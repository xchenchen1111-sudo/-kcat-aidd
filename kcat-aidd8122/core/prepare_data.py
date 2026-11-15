"""Prepare author-curated MCKcat reaction records without inventing missing products/Km."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,rdBase
from sklearn.model_selection import GroupShuffleSplit


def canonical(s):
    if not isinstance(s,str) or not s.strip(): return None
    with rdBase.BlockLogs(): m=Chem.MolFromSmiles(s)
    if m is None or not m.GetNumAtoms() or any(a.GetAtomicNum()==0 for a in m.GetAtoms()): return None
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)


def groups(df):
    parent=list(range(len(df)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for i,row in enumerate(df.itertuples()):
        keys=['seq:'+row.sequence]
        # Only recognized accessions; unresolved strings are not treated as a common protein.
        ids=re.findall(r'\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?\b',str(row.uniprot))
        keys += ['acc:'+s.split('-')[0] for s in ids]
        for key in keys:
            if key in seen: parent[find(i)]=find(seen[key])
            else: seen[key]=i
    return [str(find(i)) for i in range(len(df))]


def prepare(raw,author_test,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    df=pd.read_csv(raw);test=pd.read_csv(author_test);n=len(df)
    if df.id.duplicated().any():raise ValueError('Duplicate source IDs')
    if not set(test.id)<=set(df.id):raise ValueError('Author test IDs not found in master')
    # Check the test is really a subset of this release, not merely same IDs.
    a=df.set_index('id').loc[test.id];b=test.set_index('id')
    for col in ['sequence','reaction_smiles','log10kcat_max']:
        if not np.array_equal(a[col].astype(str).to_numpy(),b[col].astype(str).to_numpy()):
            raise ValueError('Author partition content mismatch: '+col)
    notes=[[] for _ in range(n)]
    for i,s in enumerate(df.sequence):
        if not isinstance(s,str) or not s or set(s)-set('ACDEFGHIKLMNPQRSTVWY'):notes[i].append('nonstandard_or_empty_sequence')
    y=pd.to_numeric(df.log10kcat_max,errors='coerce').to_numpy()
    k=pd.to_numeric(df.kcat,errors='coerce').to_numpy()
    for i in range(n):
        if not np.isfinite(y[i]) or not np.isfinite(k[i]) or k[i]<=0:notes[i].append('invalid_label')
        elif not np.isclose(np.log10(k[i]),y[i],atol=1e-5):notes[i].append('label_log_mismatch')
    lookup={s:canonical(s) for col in ['reactant_smiles','product'] for s in df[col].dropna().unique()}
    df['reactants_smiles']=df.reactant_smiles.map(lookup)
    df['products_smiles']=df['product'].map(lookup)
    for i,row in enumerate(df.itertuples()):
        r,p=row.reactants_smiles,row.products_smiles
        if not isinstance(r,str):notes[i].append('missing_or_invalid_reactants')
        if not isinstance(p,str):notes[i].append('missing_or_invalid_products')
        if isinstance(r,str) and isinstance(p,str):
            if r==p:notes[i].append('identical_sides')
            parts=str(row.reaction_smiles).split('>')
            if len(parts)!=3 or canonical(parts[0])!=r or canonical(parts[2])!=p:
                notes[i].append('reaction_side_mismatch')
    df['exclusion_reason']=[';'.join(x) for x in notes]
    excluded=df[df.exclusion_reason!=''].copy();keep=df[df.exclusion_reason==''].copy()
    key=['sequence','reactants_smiles','products_smiles']
    # Avoid newly selecting maxima across conflicts: exclude conflicting identical model inputs.
    spans=keep.groupby(key).log10kcat_max.transform(lambda s: s.max()-s.min())
    conflict=spans>1e-5
    bad=keep[conflict].copy();bad['exclusion_reason']='conflicting_identical_input_labels'
    excluded=pd.concat([excluded,bad]);keep=keep[~conflict].copy()
    # If a duplicate occurs in author test, preserve test membership.
    keep['author_test']=keep.id.isin(set(test.id))
    keep=keep.sort_values('author_test',ascending=False,kind='stable')
    duplicate=keep.duplicated(key,keep='first')
    bad=keep[duplicate].copy();bad['exclusion_reason']='duplicate_identical_input_same_label'
    excluded=pd.concat([excluded,bad]);keep=keep[~duplicate].sort_values('id').reset_index(drop=True)
    keep['split_group']=groups(keep)
    test_groups=set(keep.loc[keep.author_test,'split_group'])
    overlap=(~keep.author_test)&keep.split_group.isin(test_groups)
    purged=keep[overlap].copy();purged['exclusion_reason']='development_group_overlaps_author_test'
    excluded=pd.concat([excluded,purged]);keep=keep[~overlap].reset_index(drop=True)
    pool=np.flatnonzero(~keep.author_test.to_numpy());te=np.flatnonzero(keep.author_test.to_numpy())
    ti,vi=next(GroupShuffleSplit(n_splits=1,test_size=.1,random_state=20261113).split(pool,groups=keep.iloc[pool].split_group))
    tr,va=pool[ti],pool[vi]
    for x,z in [(tr,va),(tr,te),(va,te)]:
        assert not set(keep.iloc[x].split_group)&set(keep.iloc[z].split_group)
        assert not set(keep.iloc[x].sequence)&set(keep.iloc[z].sequence)
    keep['kcat_s_1']=10.**keep.log10kcat_max
    keep['ec']=keep.EC
    keep['record_id']='MCKcat:'+keep.id.astype(str)
    cols=['record_id','sequence','reactants_smiles','products_smiles','kcat_s_1','split_group','ec','uniprot','EnzymeType','id','log10kcat_max']
    keep[cols].to_csv(out/'train.csv',index=False)
    splits=pd.DataFrame(dict(row=np.arange(len(keep)),split_group=keep.split_group,split='train'))
    splits.loc[va,'split']='validation';splits.loc[te,'split']='test';splits.to_csv(out/'splits.csv',index=False)
    (out/'input_sha256.txt').write_text(hashlib.sha256((out/'train.csv').read_bytes()).hexdigest())
    excluded.to_csv(out/'excluded.csv',index=False)
    keep.iloc[tr[:3]][['sequence','reactants_smiles','products_smiles']].to_csv(out/'predict_example.csv',index=False)
    report=dict(raw_rows=n,raw_product_nonempty=int(df['product'].notna().sum()),retained_rows=len(keep),
                unique_sequences=keep.sequence.nunique(),unique_reactions=keep[['reactants_smiles','products_smiles']].drop_duplicates().shape[0],
                train=len(tr),validation=len(va),test=len(te),
                long_sequence_rows=int((keep.sequence.str.len()>1022).sum()),
                exclusion_reasons=excluded.exclusion_reason.value_counts().to_dict(),
                ec_classes=keep.ec.str.split('.').str[0].value_counts().to_dict(),
                target='author aggregated log10kcat_max; s^-1',
                split='Author sim40/N20test IDs; purge development accession/sequence groups shared with test; validation accession/sequence-group split from remaining development',
                caveats=['Author sim40 maximum pairwise identity NOT independently verified.',
                         'Validation grouping is accession/exact sequence, NOT 40% homology clustering.',
                         'Previous EC1 experiments share source data: not a fully independent external test.',
                         'No Km labels; no missing products inferred; no mass-balance cutoff.',
                         'Long sequences encoded with nonoverlapping 1022-residue windows.'])
    assert len(keep)+len(excluded)==n
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--author-test',required=True);ap.add_argument('--output',default='data')
    a=ap.parse_args();prepare(a.raw,a.author_test,a.output)

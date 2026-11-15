"""Audit existing development predictions; never use original holdout labels."""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import redox
from optimize import split_indices, digest


def summarize(frame, column):
    rows=[]
    for name, part in frame.groupby(column, observed=True):
        for pred in [c for c in part if c.startswith('pred_')]:
            rows.append(dict(group=str(name),model=pred[5:],n=len(part),
                             **redox.metrics(part.true_log10_kcat,part[pred])))
    return pd.DataFrame(rows)


def annotate_seen(pred, source, fit):
    result=pred.copy()
    seqs=set(source.loc[fit,'sequence'])
    result['enzyme_seen']=np.where(source.loc[result.source_row,'sequence'].isin(seqs).to_numpy(),
                                  'exact_sequence_seen','exact_sequence_unseen')
    return result


def main(a):
    source=pd.read_csv(a.input)
    allowed,_,_=split_indices(source,a.splits,a.input)
    run=Path(a.run); provenance=json.loads((run/'provenance.json').read_text())
    if provenance['input_sha256']!=digest(a.input) or provenance['split_sha256']!=digest(a.splits):
        raise ValueError('Benchmark/data hash mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for folder in sorted(run.glob('cold_*_*')):
        if not folder.is_dir(): continue
        parts=pd.read_csv(folder/'split.csv');pred=pd.read_csv(folder/'predictions.csv')
        if parts.source_row.duplicated().any() or set(parts.source_row)!=set(allowed):
            raise ValueError('Development pool differs from original training pool')
        fit=parts.loc[parts.partition=='fit','source_row'].to_numpy()
        dev=parts.loc[parts.partition=='development','source_row'].to_numpy()
        if set(fit)&set(dev) or set(fit)|set(dev)!=set(allowed): raise ValueError('Invalid partition')
        if pred.source_row.duplicated().any() or set(pred.source_row)!=set(dev): raise ValueError('Prediction rows mismatch')
        if not np.allclose(pred.true_log10_kcat,np.log10(source.loc[pred.source_row,'kcat_s_1'])):
            raise ValueError('Prediction labels mismatch')
        result=annotate_seen(pred,source,fit)
        dest=out/folder.name;dest.mkdir()
        seen=summarize(result,'enzyme_seen');seen.to_csv(dest/'by_seen_enzyme.csv',index=False)
        print(folder.name+' exact-sequence groups:\n'+seen.to_string(index=False),flush=True)
        # Unique sequences only; all search references belong to this scenario's fit set.
        trainseq=list(dict.fromkeys(source.loc[fit,'sequence']))
        queryseq=list(dict.fromkeys(source.loc[dev,'sequence']))
        for name,seqs in [('target',trainseq),('query',queryseq)]:
            (dest/f'{name}.fasta').write_text(''.join(f'>{name}{i}\n{s}\n' for i,s in enumerate(seqs)))
        hits=dest/'alignments.tsv'
        command=[a.mmseqs,'easy-search',str(dest/'query.fasta'),str(dest/'target.fasta'),str(hits),str(dest/'tmp'),
                 '--format-output','query,target,fident,qcov,tcov,alnlen,evalue,bits',
                 '--seq-id-mode','0','-s','7.5','--max-seqs',str(len(trainseq)),
                 '-c','0.8','--cov-mode','0','--threads',str(a.jobs)]
        print('Starting sequence search: '+folder.name,flush=True)
        subprocess.run(command,check=True)
        cols=['query','target','identity','qcov','tcov','alnlen','evalue','bits']
        h=pd.read_csv(hits,sep='\t',names=cols) if hits.stat().st_size else pd.DataFrame(columns=cols)
        if len(h) and not h[['identity','qcov','tcov']].apply(lambda c:c.between(0,1).all()).all():
            raise ValueError('Expected MMseqs fident/qcov/tcov fractions in [0,1]')
        h=h[(h.qcov>=.8)&(h.tcov>=.8)]
        best=h.sort_values(['identity','bits'],ascending=False).drop_duplicates('query')
        best.to_csv(dest/'best_detected_hits.csv',index=False)
        bestmap=dict(zip(best['query'],best.identity))
        lookup={s:bestmap.get(f'query{i}',np.nan) for i,s in enumerate(queryseq)}
        # Exact matches are certain even if heuristic search misses a hit.
        trainset=set(trainseq)
        for s in queryseq:
            if s in trainset: lookup[s]=1.
        result['best_detected_identity']=source.loc[result.source_row,'sequence'].map(lookup).to_numpy()
        result['identity_bin']=pd.cut(result.best_detected_identity,[-.00001,.3,.4,.6,.8,1.00001],
                                      labels=['<=30%','30-40%','40-60%','60-80%','80-100%']).astype(object)
        result['identity_bin']=result.identity_bin.fillna('no_qualifying_hit')
        result.to_csv(dest/'annotated_predictions.csv',index=False)
        report=summarize(result,'identity_bin');report.to_csv(dest/'by_sequence_identity.csv',index=False)
        result['seen_identity_group']=result.enzyme_seen+' / '+result.identity_bin.astype(str)
        summarize(result,'seen_identity_group').to_csv(dest/'by_seen_and_identity.csv',index=False)
        print(folder.name+' identity groups:\n'+report.to_string(index=False),flush=True)
    (out/'provenance.json').write_text(json.dumps(dict(arguments=vars(a),input_sha256=digest(a.input),
        note='MMseqs heuristic best detected alignment identity, >=80% coverage on BOTH sequences; not guaranteed exhaustive maximum. No hit does not imply identity zero. Original validation/test excluded.'),indent=2))
    print('Transfer diagnostics complete. No model retraining; original holdouts unused.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/train.csv');ap.add_argument('--splits',default='data/splits.csv')
    ap.add_argument('--run',default='runs/generalization_seed42')
    ap.add_argument('--output',default='runs/transfer_diagnostics_seed42')
    ap.add_argument('--mmseqs',default='mmseqs');ap.add_argument('--jobs',type=int,default=4)
    main(ap.parse_args())

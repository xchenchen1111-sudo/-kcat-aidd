"""Fixed-split feature ablations; original validation/test excluded."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from optimize import split_indices,protein,chemistry,digest
import redox


def sequence_features(sequences):
    aa='ACDEFGHIKLMNPQRSTVWY'
    return np.asarray([[np.log1p(len(s))]+[s.count(c)/len(s) for c in aa] for s in sequences],dtype=np.float32)


def ec_prefix(value,level):
    # Multiple/partial annotations are unknown, rather than invented categories.
    parts=str(value).strip().split('.')
    return '.'.join(parts[:level]) if len(parts)>=level and all(p.isdigit() for p in parts[:level]) else 'unknown'


def ec_features(values,fit,level):
    labels=np.array([ec_prefix(v,level) for v in values],dtype=object).reshape(-1,1)
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    enc.fit(labels[fit])
    return enc.transform(labels),enc.categories_[0].tolist()


def main(a):
    source=pd.read_csv(a.input);allowed,_,_=split_indices(source,a.splits,a.input)
    run=Path(a.run);meta=json.loads((run/'provenance.json').read_text())
    if meta['input_sha256']!=digest(a.input) or meta['split_sha256']!=digest(a.splits):raise ValueError('Source hash mismatch')
    folders=sorted(p for p in run.glob('cold_*_*') if p.is_dir())
    if not folders:raise ValueError('No completed development partitions')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    df=source.iloc[allowed].reset_index(drop=True)
    mapping={int(v):i for i,v in enumerate(allowed)}
    base=np.concatenate([protein(df,a.cache,a.device,'35M'),chemistry(df,'drfp')],axis=1)
    seq=sequence_features(df.sequence);y=np.log10(df.kcat_s_1.to_numpy(float));rows=[]
    for folder in folders:
        splits=pd.read_csv(folder/'split.csv')
        if splits.source_row.duplicated().any() or set(splits.source_row)!=set(allowed):raise ValueError('Pool mismatch')
        if set(splits.partition)!={'fit','development'}:raise ValueError('Unexpected partition')
        tr=np.array([mapping[i] for i in splits.loc[splits.partition=='fit','source_row']])
        va=np.array([mapping[i] for i in splits.loc[splits.partition=='development','source_row']])
        if set(tr)&set(va):raise ValueError('Overlap')
        seed=int(folder.name.rsplit('_',1)[1]);scenario=folder.name.rsplit('_',1)[0]
        dest=out/folder.name;dest.mkdir()
        e2,c2=ec_features(df.ec,tr,2);e3,c3=ec_features(df.ec,tr,3)
        (dest/'ec_categories.json').write_text(json.dumps({'level2':c2,'level3':c3},indent=2))
        # All variants have the same tree settings and seed as the development baseline.
        variants={'baseline':[], 'sequence':[seq], 'ec2':[e2], 'ec3':[e3],
                  'sequence_ec2':[seq,e2], 'sequence_ec3':[seq,e3]}
        seen=df.iloc[va].sequence.isin(set(df.iloc[tr].sequence)).to_numpy()
        preds=pd.DataFrame({'source_row':allowed[va],'true_log10_kcat':y[va],'exact_sequence_seen':seen})
        for name,extra in variants.items():
            print(f'Start {folder.name} {name}',flush=True)
            x=np.concatenate([base]+extra,axis=1) if extra else base
            model=ExtraTreesRegressor(n_estimators=a.trees,min_samples_leaf=2,max_features=.5,n_jobs=a.jobs,random_state=seed)
            model.fit(x[tr],y[tr]);pred=model.predict(x[va]);preds['pred_'+name]=pred
            for group,mask in [('all',np.ones(len(va),dtype=bool)),('exact_sequence_seen',seen),('exact_sequence_unseen',~seen)]:
                if mask.any():rows.append(dict(scenario=scenario,seed=seed,variant=name,group=group,n=int(mask.sum()),features=x.shape[1],**redox.metrics(y[va][mask],pred[mask])))
            pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
            preds.to_csv(dest/'predictions.csv',index=False)
            print(json.dumps(rows[-1]),flush=True)
        # Paired absolute-error difference on identical records, negative means improved.
        delta=[]
        baseline=np.abs(preds.pred_baseline.to_numpy()-y[va])
        for name in variants:
            err=np.abs(preds['pred_'+name].to_numpy()-y[va])
            delta.append({'variant':name,'MAE_difference_vs_baseline':float(np.mean(err-baseline))})
        pd.DataFrame(delta).to_csv(dest/'paired_mae_differences.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(arguments=vars(a),input_sha256=digest(a.input),split_sha256=digest(a.splits),
        note='EC encoder fit only on scenario fit rows; unseen categories map to all zero. EC features require known EC at inference. Development results only; no deployment model saved.'),indent=2))
    print('Feature ablation complete. Original validation/test NOT evaluated.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/train.csv');ap.add_argument('--splits',default='data/splits.csv')
    ap.add_argument('--run',default='runs/generalization_seed42');ap.add_argument('--output',default='runs/feature_ablation_seed42')
    ap.add_argument('--cache',default='cache');ap.add_argument('--device',default='cuda')
    ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--trees',type=int,default=300)
    main(ap.parse_args())

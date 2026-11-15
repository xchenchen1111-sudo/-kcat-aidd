"""Development grouped CV, restricted to the original training partition."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features',required=True)
    p.add_argument('--train-csv',required=True)
    p.add_argument('--out',default='group_cv_results')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    d=pd.read_csv(a.train_csv,dtype={'split_group':str})
    with np.load(a.features,allow_pickle=False) as f:
        x=f['X_train']; ids=f['train_ids'].astype(str); dim=int(f['protein_dim'])
    assert np.array_equal(ids,d.record_id.astype(str)) and len(x)==len(d)
    assert d.record_id.is_unique and not d.split_group.isna().any()
    assert d.groupby('sequence').split_group.nunique().max()==1
    assert 2<=a.folds<=d.split_group.nunique()
    summaries=[]; predictions=[]
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=a.folds).split(x,groups=d.split_group)):
        folder=out/f'fold_{fold}';folder.mkdir()
        d.iloc[tr].to_csv(folder/'train.csv',index=False)
        d.iloc[va].to_csv(folder/'validation.csv',index=False)
        np.savez_compressed(folder/'features.npz',X_train=x[tr],X_val=x[va],train_ids=ids[tr],val_ids=ids[va],protein_dim=dim)
        cmd=[sys.executable,str(Path(__file__).with_name('train_fusion.py')),'--features',str(folder/'features.npz'),'--train-csv',str(folder/'train.csv'),'--val-csv',str(folder/'validation.csv'),'--out',str(folder/'models'),'--epochs',str(a.epochs),'--device',a.device,'--seeds',*map(str,a.seeds)]
        print('Starting fold',fold,flush=True);subprocess.run(cmd,check=True)
        table=pd.read_csv(folder/'models/summary.csv');table['fold']=fold;summaries.append(table)
        pd.concat(summaries).to_csv(out/'fold_metrics.csv',index=False)
        for model in ['fusion_control','fusion_dropout']:
            q=pd.read_csv(folder/'models'/f'{model}_ensemble_predictions.csv');q['model']=model;q['fold']=fold;predictions.append(q)
        pd.concat(predictions).to_csv(out/'development_oof_predictions.csv',index=False)
        # Intermediate feature copies can be regenerated; retain source and trained outputs.
        (folder/'features.npz').unlink()
    from train_fusion import metrics
    allpred=pd.concat(predictions)
    report=[]
    for model,q in allpred.groupby('model'):
        assert len(q)==len(d) and q.record_id.is_unique
        report.append(dict(model=model,**metrics(q.true_log10_kcat.to_numpy(),q.pred_log10_kcat.to_numpy())))
    pd.DataFrame(report).to_csv(out/'development_summary.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(arguments=vars(a),grouping='Existing accession/exact-sequence split_group; NOT homology clustering',limitation='Fold validation is used for early stopping. Scores are development estimates, not unbiased outer-CV/test estimates.'),indent=2))
    print(pd.DataFrame(report).to_string(index=False),flush=True)

if __name__=='__main__': main()

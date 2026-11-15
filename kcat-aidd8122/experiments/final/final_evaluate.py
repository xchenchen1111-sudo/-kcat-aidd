"""Locked protocol: train-only fitting, validation stopping, one held-out evaluation."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,json,sys,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='core')
    p.add_argument('--cache',default='cache')
    p.add_argument('--out',default='final_results')
    a=p.parse_args(); project=Path(a.project);out=Path(a.out)
    if out.exists(): raise FileExistsError('Use a fresh output directory: '+str(out))
    if shutil.disk_usage(out.parent).free<8*1024**3: raise RuntimeError('Need at least 8 GiB free')
    sys.path.insert(0,str(project.resolve()))
    import optimize, neural, redox
    inp=project/'data/train.csv'; splits=project/'data/splits.csv'
    df=pd.read_csv(inp);redox.validate(df,True)
    tr,va,te=optimize.split_indices(df,splits,inp)
    original=json.loads((project/'runs/mlp35_seed42/config.json').read_text())
    assert optimize.digest(inp)==original['input_sha256'], 'Original input hash mismatch'
    assert optimize.digest(splits)==original['split_sha256'], 'Original split hash mismatch'
    out.mkdir(parents=True)
    protocol=dict(input_sha256=optimize.digest(inp),split_sha256=optimize.digest(splits),
        seeds=[42,43,44],protein_sizes=['35M','650M'],lr=.001,weight_decay=.0001,
        epochs=200,patience=20,batch_size=64,pooling=original['protein_pooling'],
        selection='Validation RMSE only',mix='Equal mean of six log10 predictions',
        counts=dict(train=len(tr),validation=len(va),test=len(te)))
    (out/'locked_protocol.json').write_text(json.dumps(protocol,indent=2))
    # Original trainer constructs features for all supplied rows. Supply only train
    # and validation, and create an explicit split callback so test stays untouched.
    dev_ids=np.concatenate([tr,va]);dev=df.iloc[dev_ids].copy()
    dev_path=out/'development.csv';dev.to_csv(dev_path,index=False)
    real_split=neural.split_indices
    neural.split_indices=lambda *_:(np.arange(len(tr)),np.arange(len(tr),len(dev)),np.array([],dtype=int))
    try:
        for size in ['35M','650M']:
            for seed in [42,43,44]:
                print('Training',size,seed,flush=True)
                args=SimpleNamespace(input=str(dev_path),splits=str(splits),output=str(out/f'{size}_seed{seed}'),cache=a.cache,device='cuda',protein=size,seed=seed,epochs=200,patience=20,batch_size=64,jobs=4)
                neural.train(args)
    finally: neural.split_indices=real_split
    # Selection is complete before any held-out features or predictions are made.
    y=np.log10(df.iloc[te].kcat_s_1.to_numpy(float));predictions={}
    for size in ['35M','650M']:
        args=SimpleNamespace(cache=a.cache,device='cuda',protein=size)
        x=optimize.feature_matrix(df.iloc[te],args,'ordered')
        seeds=[]
        for seed in [42,43,44]:
            predict,_=neural.load_predictor(out/f'{size}_seed{seed}/model.pt','cuda')
            seeds.append(predict(x))
        predictions[size]=np.mean(seeds,axis=0)
        del x
    predictions['equal_mix']=(predictions['35M']+predictions['650M'])/2
    report=[]
    for name,pred in predictions.items():
        report.append(dict(model=name,**redox.metrics(y,pred)))
        optimize.diagnostics(df,te,np.log10(df.kcat_s_1.to_numpy(float)),pred,out,'test_'+name)
    pd.DataFrame(report).to_csv(out/'test_metrics.csv',index=False)
    tails=[]
    for name,pred in predictions.items():
        for lo,hi in [(-np.inf,-2),(-2,0),(0,2),(2,4),(4,np.inf)]:
            mask=(y>=lo)&(y<hi)
            if mask.any(): tails.append(dict(model=name,lower=lo,upper=hi,n=int(mask.sum()),**redox.metrics(y[mask],pred[mask])))
    pd.DataFrame(tails).to_csv(out/'test_tail_metrics.csv',index=False)
    # Paired group bootstrap: conditional on the trained models and this split.
    groups=df.iloc[te].split_group.astype(str).to_numpy()
    _,gi=np.unique(groups,return_inverse=True);ng=gi.max()+1
    count=np.bincount(gi);rng=np.random.default_rng(20261115)
    sums={name:(np.bincount(gi,weights=(pred-y)**2),np.bincount(gi,weights=np.abs(pred-y))) for name,pred in predictions.items()}
    deltas=[]
    for _ in range(2000):
        idx=rng.integers(0,ng,size=ng);n=count[idx].sum()
        s0,a0=sums['35M'];s1,a1=sums['equal_mix']
        deltas.append([np.sqrt(s1[idx].sum()/n)-np.sqrt(s0[idx].sum()/n),(a1[idx].sum()-a0[idx].sum())/n])
    ci=np.quantile(deltas,[.025,.975],axis=0)
    (out/'paired_group_bootstrap.json').write_text(json.dumps(dict(comparison='equal_mix minus 35M; negative favors mix',groups=int(ng),replicates=2000,RMSE_95CI=ci[:,0].tolist(),MAE_95CI=ci[:,1].tolist()),indent=2))
    print(pd.DataFrame(report).to_string(index=False),flush=True)
    print('FINAL EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()

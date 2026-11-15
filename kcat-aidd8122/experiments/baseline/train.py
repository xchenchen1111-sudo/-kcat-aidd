"""Leakage-controlled comparisons on precomputed enzyme/reaction features."""
import argparse, copy, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y, p):
    e = p-y
    return dict(n=len(y), RMSE=float(np.sqrt(mean_squared_error(y,p))),
                MAE=float(mean_absolute_error(y,p)), R2=float(r2_score(y,p)) if len(y)>1 else None,
                within_10fold=float(np.mean(np.abs(e)<=1)), bias=float(e.mean()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='NPZ: X_train, X_val, train_ids, val_ids')
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--out', default='kcat_results')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tables=[pd.read_csv(args.train_csv),pd.read_csv(args.val_csv)]
    with np.load(args.features,allow_pickle=False) as f:
        xs=[np.asarray(f[k],dtype=np.float32) for k in ['X_train','X_val']]
        ids=[f[k].astype(str) for k in ['train_ids','val_ids']]
        protein_dim=int(f['protein_dim']) if 'protein_dim' in f else None
    for d,x,i in zip(tables,xs,ids):
        assert d.record_id.is_unique, 'CSV record_id must be unique'
        assert x.ndim==2 and len(x)==len(d)==len(i)
        assert np.array_equal(d.record_id.astype(str).values,i), 'Feature/CSV row order mismatch'
        assert np.isfinite(x).all(), 'Nonfinite features'
    assert xs[0].shape[1]==xs[1].shape[1]
    for key in ['record_id','sequence','split_group']:
        if all(key in d for d in tables):
            assert not set(tables[0][key].dropna()) & set(tables[1][key].dropna()), f'Split overlap: {key}'
    ys=[d.true_log10_kcat.to_numpy(dtype=np.float32) for d in tables]
    assert all(np.isfinite(y).all() for y in ys)
    if protein_dim is not None:
        for x in xs:
            assert np.all(x[:,protein_dim:]>=0)
            x[:,protein_dim:]=np.log1p(x[:,protein_dim:])
    scaler=StandardScaler().fit(xs[0]); xs=[scaler.transform(x).astype(np.float32) for x in xs]
    ym,yscale=float(ys[0].mean()),float(ys[0].std())
    assert yscale>0
    np.savez(out/'normalization.npz',feature_mean=scaler.mean_,feature_scale=scaler.scale_,target_mean=ym,target_scale=yscale,protein_dim=protein_dim if protein_dim is not None else -1)
    summary=[]; bins=[]
    def save(name,p,extra=None):
        summary.append(dict(model=name,**metrics(ys[1],p),**(extra or {})))
        pd.DataFrame(dict(record_id=ids[1],true_log10_kcat=ys[1],pred_log10_kcat=p)).to_csv(out/(name+'_predictions.csv'),index=False)
        for lo,hi in [(-np.inf,-2),(-2,0),(0,2),(2,4),(4,np.inf)]:
            mask=(ys[1]>=lo)&(ys[1]<hi)
            if mask.any(): bins.append(dict(model=name,lower=lo,upper=hi,**metrics(ys[1][mask],p[mask])))
        pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
        pd.DataFrame(bins).to_csv(out/'tail_metrics.csv',index=False)
        print(summary[-1],flush=True)
    save('mean',np.full(len(ys[1]),ym))
    for alpha in [1.,100.,10000.]:
        reg=Ridge(alpha=alpha).fit(xs[0],ys[0]); save(f'ridge_{alpha:g}',reg.predict(xs[1]))
        np.savez(out/f'ridge_{alpha:g}.npz',coef=reg.coef_,intercept=reg.intercept_)
    # A small sequential ablation, rather than a large hyperparameter search.
    configs=[dict(name='small',drop=0.2,wd=1e-3,lr=3e-4,weighted=False),
             dict(name='regularized',drop=0.4,wd=1e-2,lr=3e-4,weighted=False),
             dict(name='tail_weighted',drop=0.4,wd=1e-2,lr=3e-4,weighted=True)]
    for cfg in configs:
        predictions=[]
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            net=nn.Sequential(nn.Linear(xs[0].shape[1],256),nn.ReLU(),nn.Dropout(cfg['drop']),nn.Linear(256,64),nn.ReLU(),nn.Dropout(cfg['drop']),nn.Linear(64,1)).to(args.device)
            opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
            weights=np.ones(len(ys[0]),dtype=np.float32)
            if cfg['weighted']:
                labels=np.digitize(ys[0],[-2,0,2,4]); counts=np.bincount(labels,minlength=5)
                weights=np.sqrt(len(labels)/(5*np.maximum(counts[labels],1))).clip(.5,3).astype(np.float32)
                weights/=weights.mean()
            ds=torch.utils.data.TensorDataset(torch.from_numpy(xs[0]),torch.from_numpy((ys[0]-ym)/yscale),torch.from_numpy(weights))
            loader=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True)
            def predict(x):
                net.eval()
                with torch.no_grad(): return np.concatenate([net(torch.from_numpy(z).to(args.device)).flatten().cpu().numpy() for z in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))])*yscale+ym
            best=float('inf'); best_state=None; stale=0; history=[]
            for epoch in range(1,args.epochs+1):
                net.train(); total=0
                for x,y,w in loader:
                    x,y,w=[v.to(args.device) for v in (x,y,w)]
                    loss=((net(x).flatten()-y).square()*w).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5); opt.step(); total+=loss.item()*len(x)
                score=metrics(ys[1],predict(xs[1]))['RMSE']
                history.append(dict(epoch=epoch,train_loss=total/len(ds),val_RMSE=score))
                if score<best:
                    best=score; best_state=copy.deepcopy(net.state_dict()); best_epoch=epoch; stale=0
                else: stale+=1
                if stale>=args.patience: break
            assert best_state is not None, 'Training failed: inspect nonfinite loss'
            net.load_state_dict(best_state); name=f"{cfg['name']}_seed{seed}"
            p=predict(xs[1]); predictions.append(p)
            save(name,p,dict(best_epoch=best_epoch,train_RMSE=metrics(ys[0],predict(xs[0]))['RMSE']))
            torch.save(dict(state_dict={k:v.cpu() for k,v in best_state.items()},input_dim=xs[0].shape[1],config=cfg,seed=seed),out/(name+'.pt'))
            pd.DataFrame(history).to_csv(out/(name+'_history.csv'),index=False)
        save(cfg['name']+'_ensemble',np.mean(predictions,axis=0))
    (out/'run_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')

if __name__=='__main__': main()

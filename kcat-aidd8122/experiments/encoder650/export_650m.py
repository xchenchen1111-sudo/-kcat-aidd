"""Training-only export: preserve reaction features and sample order from 35M."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='core')
    p.add_argument('--cache',default='cache')
    p.add_argument('--train-csv',default='core/runs/mlp35_seed42/train_predictions.csv')
    p.add_argument('--source',default='features/features35.npz')
    p.add_argument('--out',default='features_650m_train.npz')
    a=p.parse_args(); target=Path(a.out)
    if target.exists(): raise FileExistsError(target)
    if shutil.disk_usage(target.parent).free<8*1024**3: raise RuntimeError('Need at least 8 GiB free before export and CV')
    d=pd.read_csv(a.train_csv)
    with np.load(a.source,allow_pickle=False) as f:
        ids=f['train_ids'].astype(str); old=f['X_train']; dim=int(f['protein_dim'])
    assert np.array_equal(ids,d.record_id.astype(str)) and dim==480
    assert old.shape==(len(d),dim+4096)
    root=Path(a.cache); namespace='hf_esm2_t33_650M_UR50D_v1'
    counts=dict(window_cache=0,short_legacy_cache=0,compute=0)
    for seq in d.sequence.unique():
        name=hashlib.sha256(seq.encode()).hexdigest()+'.npy'
        window=root/(namespace+'_window1022')/name; legacy=root/namespace/name
        path=window if window.exists() else legacy if len(seq)<=1022 and legacy.exists() else None
        if path:
            v=np.load(path,allow_pickle=False)
            if v.shape!=(1280,) or not np.isfinite(v).all(): raise ValueError('Invalid cache: '+str(path))
            counts['window_cache' if path==window else 'short_legacy_cache']+=1
        else: counts['compute']+=1
    print('Unique sequence cache audit:',counts,flush=True)
    sys.path.insert(0,str(Path(a.project).resolve()))
    import optimize
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Requested GPU is unavailable')
    # Run with CUDA_VISIBLE_DEVICES=1, so original engine can use cuda.
    protein=optimize.protein(d,a.cache,'cuda','650M')
    x=np.concatenate([protein,old[:,dim:]],axis=1).astype(np.float32)
    assert np.isfinite(x).all()
    temporary=target.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,X_train=x,train_ids=ids,protein_dim=1280)
    temporary.replace(target)
    target.with_suffix('.audit.json').write_text(json.dumps(dict(cache=counts,n=len(d),shape=list(x.shape),chemistry='Copied unchanged from 35M feature archive',pooling='Original reaction project length-weighted nonoverlapping 1022 windows',scope='Original training partition only; validation and test not extracted',source=a.source),indent=2))
    print('Export complete:',target,flush=True)

if __name__=='__main__': main()

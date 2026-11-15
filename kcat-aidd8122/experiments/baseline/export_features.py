"""Export original ordered features and verify them against saved predictions."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='core')
    p.add_argument('--run',default=None)
    p.add_argument('--cache',default='cache')
    p.add_argument('--device',default='auto')
    p.add_argument('--output',default='features.npz')
    a=p.parse_args()
    project=Path(a.project).resolve(); run=Path(a.run) if a.run else project/'runs/mlp35_seed42'
    sys.path.insert(0,str(project))
    import optimize, neural
    config=json.loads((run/'config.json').read_text())
    if config.get('protein_pooling')!='length_weighted_nonoverlapping_1022_residue_windows':
        raise ValueError('Unexpected pooling configuration; inspect original run first')
    if config['representation']!='ordered': raise ValueError('Expected ordered representation')
    predictor,_=neural.load_predictor(run/'model.pt',a.device)
    args=SimpleNamespace(cache=a.cache,device=a.device,protein=config['protein'])
    payload={}; report={}
    for split,short in [('train','train'),('validation','val')]:
        df=pd.read_csv(run/f'{split}_predictions.csv')
        if not df.record_id.is_unique: raise ValueError('Duplicate record IDs')
        x=optimize.feature_matrix(df,args,'ordered')
        pred=predictor(x)
        delta=np.abs(pred-df.pred_log10_kcat.to_numpy())
        report[split]=dict(n=len(df),max_prediction_difference=float(delta.max()),mean_prediction_difference=float(delta.mean()))
        print(report[split],flush=True)
        if not np.isfinite(x).all() or not np.allclose(pred,df.pred_log10_kcat.to_numpy(),atol=1e-3,rtol=1e-4):
            raise ValueError('Original predictions not reproduced: check cache, source version and run. No export saved.')
        payload['X_'+short]=x.astype(np.float32)
        payload[short+'_ids']=df.record_id.astype(str).to_numpy(dtype=str)
    payload['protein_dim']=np.array(480 if config['protein']=='35M' else 1280)
    target=Path(a.output)
    if target.exists(): raise FileExistsError(target)
    np.savez_compressed(target,**payload)
    target.with_suffix('.audit.json').write_text(json.dumps(dict(project=str(project),run=str(run),cache=a.cache,config=config,reproduction=report),indent=2))
    print('Verified features saved:',target,flush=True)

if __name__=='__main__': main()

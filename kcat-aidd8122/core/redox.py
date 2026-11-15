import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import importlib.metadata as metadata

MODEL='hf_esm2_t12_35M_UR50D_v1'
HF_MODEL='facebook/esm2_t12_35M_UR50D'
AA=set('ACDEFGHIKLMNPQRSTVWY')
def validate(df,training=False):
    cols=['sequence','reactants_smiles','products_smiles']+(['kcat_s_1','split_group'] if training else [])
    missing=set(cols)-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {missing}')
    if df.empty or df[cols].isna().any().any(): raise ValueError('Empty data or missing required inputs')
    for i,s in enumerate(df.sequence):
        if not isinstance(s,str) or not s or set(s)-AA: raise ValueError(f'Invalid sequence at CSV row {i+2}; nonempty standard amino acids required')
    if training:
        v=pd.to_numeric(df.kcat_s_1,errors='coerce').to_numpy()
        if not np.isfinite(v).all() or (v<=0).any(): raise ValueError('kcat_s_1 must be finite and positive')
        if df.groupby('sequence').split_group.nunique().max()>1: raise ValueError('Same sequence crosses groups')
        if df.split_group.nunique()<5: raise ValueError('At least five sequence clusters required')

def molecular_features(df):
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    blocks=[]
    for col in ['reactants_smiles','products_smiles']:
        cache={}
        for s in df[col].unique():
            with rdBase.BlockLogs(): m=Chem.MolFromSmiles(s)
            if m is None or not m.GetNumAtoms() or any(a.GetAtomicNum()==0 for a in m.GetAtoms()): raise ValueError(f'Invalid molecule in {col}: {s}')
            cache[s]=gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        blocks.append(np.stack([cache[s] for s in df[col]]))
    return np.concatenate(blocks,axis=1)

def protein_features(sequences,cache,device='auto',feature_dim=480):
    import torch
    from transformers import AutoTokenizer, EsmModel
    cache_root=Path(cache)
    cache=cache_root/(MODEL+'_window1022');cache.mkdir(parents=True,exist_ok=True)
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    model=None;result={}
    for i,seq in enumerate(dict.fromkeys(sequences)):
        p=cache/(hashlib.sha256(seq.encode()).hexdigest()+'.npy')
        legacy=cache_root/MODEL/p.name
        if p.exists(): v=np.load(p,allow_pickle=False)
        elif len(seq)<=1022 and legacy.exists(): v=np.load(legacy,allow_pickle=False)
        else:
            if model is None:
                tokenizer=AutoTokenizer.from_pretrained(HF_MODEL)
                model=EsmModel.from_pretrained(HF_MODEL,add_pooling_layer=False,use_safetensors=True).eval().to(dev)
            v=np.zeros(feature_dim,dtype=np.float64)
            for start in range(0,len(seq),1022):
                part=seq[start:start+1022]
                tokens=tokenizer(part,return_tensors='pt',return_special_tokens_mask=True)
                mask=tokens.pop('special_tokens_mask').to(dev)==0
                tokens={k:v.to(dev) for k,v in tokens.items()}
                mask=mask & tokens['attention_mask'].bool()
                if int(mask.sum())!=len(part): raise ValueError('Tokenizer residue count mismatch')
                with torch.inference_mode():
                    v+=model(**tokens).last_hidden_state[0,mask[0]].float().mean(0).cpu().numpy()*len(part)
            v=(v/len(seq)).astype(np.float32)
            temp=p.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,v)
            temp.replace(p)
        if v.shape!=(feature_dim,) or not np.isfinite(v).all():raise ValueError(f'Invalid cache file: {p}')
        result[seq]=v
        if (i+1)%50==0:print(f'Protein features: {i+1} unique sequences',flush=True)
    return np.stack([result[s] for s in sequences])

def features(df,cache,device):
    chemistry=molecular_features(df)
    protein=protein_features(df.sequence.tolist(),cache,device)
    return np.concatenate([protein,chemistry],axis=1)

def metrics(y,p):
    return {'MAE_log10':float(mean_absolute_error(y,p)),'RMSE_log10':float(np.sqrt(mean_squared_error(y,p))),'R2_log10':float(r2_score(y,p)) if len(y)>1 else None,'within_10fold':float(np.mean(abs(y-p)<=1))}

def fit_model(df,x,out,seed=42,trees=300,jobs=-1):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    y=np.log10(df.kcat_s_1.to_numpy(dtype=float));groups=df.split_group.astype(str).to_numpy()
    pool,te=next(GroupShuffleSplit(n_splits=1,test_size=.1,random_state=seed).split(x,y,groups))
    ti,vi=next(GroupShuffleSplit(n_splits=1,test_size=1/9,random_state=seed+1).split(x[pool],y[pool],groups[pool]))
    tr,va=pool[ti],pool[vi]
    assert not set(groups[tr])&set(groups[te])
    assert not set(groups[tr])&set(groups[va])
    assert not set(groups[va])&set(groups[te])
    model=ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=2,max_features=.5,random_state=seed,n_jobs=jobs)
    print(f'Train/validation/test rows: {len(tr)}/{len(va)}/{len(te)}',flush=True)
    model.fit(x[tr],y[tr]);pred=model.predict(x[te])
    report={'validation':metrics(y[va],model.predict(x[va])),'validation_rows':len(va),'validation_groups':len(set(groups[va])),'test':metrics(y[te],pred),'median_baseline':metrics(y[te],np.full(len(te),np.median(y[tr]))),'train_rows':len(tr),'test_rows':len(te),'train_groups':len(set(groups[tr])),'test_groups':len(set(groups[te])),'seed':seed}
    config={'schema_version':1,'protein_model':MODEL,'pooling':'mean_non_special_tokens','fingerprint':'Morgan_count','radius':2,'fp_size':2048,'feature_order':['protein','reactants','products'],'feature_dim':4576,'target':'log10(kcat / s^-1)','seed':seed,'trees':trees,'python':platform.python_version(),'versions':{p:metadata.version(p) for p in ['numpy','pandas','rdkit','scikit-learn','joblib']}}
    joblib.dump({'model':model,'config':config},out/'model.joblib')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8');(out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    split=pd.DataFrame({'row':np.arange(len(df)),'split_group':groups,'split':'train'});split.loc[va,'split']='validation';split.loc[te,'split']='test';split.to_csv(out/'splits.csv',index=False)
    p=df.iloc[te][['sequence','reactants_smiles','products_smiles','kcat_s_1']].copy();p['pred_log10_kcat']=pred;p['pred_kcat_s_1']=10**pred;p.to_csv(out/'test_predictions.csv',index=False)
    return report

def main():
    ap=argparse.ArgumentParser(description='EC1 kcat: frozen ESM2 + ordered molecular fingerprints + ExtraTrees')
    sub=ap.add_subparsers(dest='command',required=True)
    for command in ['train','predict']:
        p=sub.add_parser(command);p.add_argument('--input',required=True);p.add_argument('--cache',default='cache');p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if command=='train':
            p.add_argument('--output',default='runs/baseline');p.add_argument('--seed',type=int,default=42);p.add_argument('--trees',type=int,default=300);p.add_argument('--jobs',type=int,default=-1)
        else:p.add_argument('--model',required=True);p.add_argument('--output',default='predictions.csv')
    a=ap.parse_args();df=pd.read_csv(a.input);validate(df,a.command=='train')
    bundle=None
    if a.command=='predict':
        bundle=joblib.load(a.model)
        if bundle['config']['protein_model']!=MODEL or bundle['config']['feature_dim']!=4576:raise ValueError('Incompatible model schema')
        for package,version in bundle['config']['versions'].items():
            if metadata.version(package)!=version:raise ValueError(f'Use training version: {package}=={version}')
    x=features(df,a.cache,a.device)
    if a.command=='train':
        report=fit_model(df,x,a.output,a.seed,a.trees,a.jobs)
        Path(a.output,'input_sha256.txt').write_text(hashlib.sha256(Path(a.input).read_bytes()).hexdigest())
        print(json.dumps(report,indent=2))
    else:
        pred=bundle['model'].predict(x);df['pred_log10_kcat']=pred;df['pred_kcat_s_1']=10**pred
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.output,index=False);print(f'Saved {len(df)} predictions to {a.output}')
if __name__=='__main__':main()

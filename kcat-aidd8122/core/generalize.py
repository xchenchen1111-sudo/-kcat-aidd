"""Training-pool-only cold enzyme/reaction benchmark. Original validation/test stay sealed."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import normalize
import redox
from optimize import split_indices, protein, chemistry, digest


def reaction_key(r, p):
    # Group forward and reverse together; prediction features retain their original direction.
    return '>>'.join(sorted([r, p]))


def group_split(groups, seed):
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(groups, groups=groups))
    assert not set(groups[tr]) & set(groups[va])
    return tr, va


def cluster_groups(df, table):
    parent = list(range(len(df)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in df.iterrows():
        for key in [('accession_group', str(row.split_group)), ('cluster', table[f's{i}'])]:
            if key in seen: parent[root(i)] = root(seen[key])
            else: seen[key] = i
    return np.array([str(root(i)) for i in range(len(df))])


def neighbors(p, r, train, query, y, k=16, alpha=4., beta=4.):
    """Cosine ESM + binary DRFP Tanimoto, chunked; references are train only."""
    pp = normalize(p); rb = (r > 0).astype(np.float32)
    pred=[]; ps=[]; rs=[]
    k=min(k,len(train))
    for begin in range(0,len(query),64):
        q=query[begin:begin+64]
        sp=np.clip(pp[q] @ pp[train].T,-1,1)
        inter=rb[q] @ rb[train].T
        union=rb[q].sum(1)[:,None]+rb[train].sum(1)[None,:]-inter
        sr=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        score=alpha*sp+beta*sr
        idx=np.argpartition(score,-k,axis=1)[:,-k:]
        logits=np.take_along_axis(score,idx,axis=1); logits-=logits.max(1,keepdims=True)
        weights=np.exp(logits); weights/=weights.sum(1,keepdims=True)
        pred.extend((weights*y[train][idx]).sum(1)); ps.extend(sp.max(1)); rs.extend(sr.max(1))
    return np.array(pred),np.array(ps),np.array(rs)


def main(a):
    df=pd.read_csv(a.input)
    tr,_,_=split_indices(df,a.splits,a.input)
    # Nothing downstream reads original holdout labels/features.
    df=df.iloc[tr].copy().reset_index(drop=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    df['source_row']=tr
    fasta=out/'train_sequences.fasta'
    fasta.write_text(''.join(f'>s{i}\n{s}\n' for i,s in enumerate(df.sequence)),encoding='utf8')
    subprocess.run([a.mmseqs,'easy-cluster',str(fasta),str(out/'clusters'),str(out/'mmseqs_tmp'),
                    '--min-seq-id','0.4','-c','0.8','--cov-mode','0','--threads',str(a.jobs)],check=True)
    ct=pd.read_csv(out/'clusters_cluster.tsv',sep='\t',header=None,names=['representative','member'])
    table=dict(zip(ct.member,ct.representative))
    if set(table)!=set(f's{i}' for i in range(len(df))): raise ValueError('Incomplete cluster membership')
    eg=cluster_groups(df,table)
    rg=np.array([reaction_key(r,p) for r,p in zip(df.reactants_smiles,df.products_smiles)])
    p=protein(df,a.cache,a.device,'35M');r=chemistry(df,'drfp')
    x=np.concatenate([p,r],axis=1);y=np.log10(df.kcat_s_1.to_numpy(float))
    results=[]
    for scenario,groups in [('cold_enzyme',eg),('cold_reaction',rg)]:
        for seed in a.seeds:
            train,dev=group_split(groups,seed)
            it,iv=group_split(groups[train],seed+1000); inner_train=train[it];inner_val=train[iv]
            folder=out/f'{scenario}_{seed}';folder.mkdir()
            assignment=np.full(len(df),'fit',dtype=object);assignment[dev]='development'
            pd.DataFrame({'source_row':tr,'group':groups,'partition':assignment}).to_csv(folder/'split.csv',index=False)
            # Tune only on inner validation, then evaluate once on development.
            scores=[]
            for k,alpha,beta in [(8,4.,4.),(32,4.,4.),(16,8.,2.),(16,2.,8.)]:
                pred,_,_=neighbors(p,r,inner_train,inner_val,y,k,alpha,beta)
                scores.append((redox.metrics(y[inner_val],pred)['RMSE_log10'],k,alpha,beta))
            _,k,alpha,beta=min(scores)
            knn,ps,rs=neighbors(p,r,train,dev,y,k,alpha,beta)
            pd.DataFrame(scores,columns=['inner_rmse','k','alpha','beta']).to_csv(folder/'neighbor_selection.csv',index=False)
            predictions={'median':np.full(len(dev),np.median(y[train])),'neighbors':knn}
            # Fixed ExtraTrees reference, inner-validation CatBoost early stopping.
            model=ExtraTreesRegressor(n_estimators=a.trees,min_samples_leaf=2,max_features=.5,n_jobs=a.jobs,random_state=seed)
            model.fit(x[train],y[train]);predictions['extra']=model.predict(x[dev])
            from catboost import CatBoostRegressor
            params=dict(iterations=a.trees,depth=6,learning_rate=.05,l2_leaf_reg=10,
                        thread_count=a.jobs,random_seed=seed,verbose=False,allow_writing_files=False)
            cb=CatBoostRegressor(**params)
            cb.fit(x[inner_train],y[inner_train],eval_set=(x[inner_val],y[inner_val]),early_stopping_rounds=40)
            params['iterations']=max(1,cb.tree_count_)
            cb=CatBoostRegressor(**params);cb.fit(x[train],y[train]);predictions['catboost']=cb.predict(x[dev])
            base=pd.DataFrame({'source_row':tr[dev],'true_log10_kcat':y[dev],
                               'max_esm_cosine':ps,'max_drfp_tanimoto':rs})
            bins=[]
            # These cosine bins are NOT sequence identity percentages.
            for family,pred in predictions.items():
                results.append(dict(scenario=scenario,seed=seed,model=family,n=len(dev),**redox.metrics(y[dev],pred)))
                base[f'pred_{family}']=pred
                for axis,values,edges in [('esm_cosine',ps,[-1,.8,.9,.95,1.00001]),('drfp_tanimoto',rs,[-.00001,.25,.5,.75,1.00001])]:
                    labels=pd.cut(values,edges,include_lowest=True)
                    for label in labels.categories:
                        mask=np.asarray(labels==label)
                        if mask.any(): bins.append(dict(model=family,axis=axis,bin=str(label),n=int(mask.sum()),**redox.metrics(y[dev][mask],pred[mask])))
            base.to_csv(folder/'predictions.csv',index=False)
            pd.DataFrame(bins).to_csv(folder/'similarity_errors.csv',index=False)
            pd.DataFrame(results).to_csv(out/'results.csv',index=False)
            print(json.dumps(results[-4:],ensure_ascii=False),flush=True)
    (out/'provenance.json').write_text(json.dumps(dict(input_sha256=digest(a.input),split_sha256=digest(a.splits),
        arguments=vars(a),original_training_rows=len(df),note='Original validation/test unused. MMseqs representative clusters do not prove a strict cross-split identity ceiling. ESM cosine is not sequence identity. Reaction split allows seen enzymes.'),indent=2),encoding='utf8')
    print('Development benchmark complete. Original validation and test NOT evaluated.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='data/train.csv');ap.add_argument('--splits',default='data/splits.csv')
    ap.add_argument('--output',default='runs/generalization_v1');ap.add_argument('--cache',default='cache')
    ap.add_argument('--device',default='cuda');ap.add_argument('--mmseqs',default='mmseqs')
    ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--trees',type=int,default=300)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    main(ap.parse_args())

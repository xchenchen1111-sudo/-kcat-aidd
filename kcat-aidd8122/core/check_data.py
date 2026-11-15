import json
from pathlib import Path
import numpy as np
import pandas as pd
import redox
from optimize import split_indices,chemistry

root=Path(__file__).resolve().parent
df=pd.read_csv(root/'data/train.csv');redox.validate(df,True)
tr,va,te=split_indices(df,root/'data/splits.csv',root/'data/train.csv')
assert not df.duplicated(['sequence','reactants_smiles','products_smiles']).any()
for ids in [tr,va,te]: assert len(ids)>0
print('Rows:',len(df),'train/validation/test:',len(tr),len(va),len(te))
print('Unique sequences:',df.sequence.nunique(),'EC classes:',sorted(df.ec.str.split('.').str[0].unique()))
for rep in ['reactants','ordered','ordered_diff','drfp']:
    x=chemistry(df.head(3),rep);assert np.isfinite(x).all();print(rep,x.shape)
print('PASS: data hash, partition isolation, duplicate checks, feature smoke tests; no ESM inference.')

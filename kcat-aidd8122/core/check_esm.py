import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from optimize import feature_matrix

ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args()
df=pd.read_csv(Path(__file__).resolve().parent/'data/train.csv')
sample=pd.concat([df.iloc[:1],df[df.sequence.str.len()>1022].iloc[:1]])
cfg=SimpleNamespace(cache='cache_smoke',device=a.device,protein='35M')
x=feature_matrix(sample,cfg,'ordered')
assert x.shape==(len(sample),4576) and np.isfinite(x).all()
print('PASS: real ESM2/RDKit including long sequence; lengths:',sample.sequence.str.len().tolist())

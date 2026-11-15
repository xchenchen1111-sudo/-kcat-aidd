"""Verify window coverage/weighting/cache using a deterministic fake ESM encoder."""
import sys,types,tempfile
from pathlib import Path
import torch
import numpy as np
import redox


def main():
    calls=[]
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,*args,**kw):return cls()
        def __call__(self,seq,**kw):
            calls.append(len(seq));v=[0]+[1 if a=='A' else 3 for a in seq]+[0]
            return dict(input_ids=torch.tensor([v]),attention_mask=torch.ones(1,len(v),dtype=torch.long),
                        special_tokens_mask=torch.tensor([[1]+[0]*len(seq)+[1]]))
    class Model:
        @classmethod
        def from_pretrained(cls,*args,**kw):return cls()
        def eval(self):return self
        def to(self,dev):return self
        def __call__(self,input_ids,**kw):
            return types.SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1,1,480))
    fake=types.ModuleType('transformers');fake.AutoTokenizer=Tokenizer;fake.EsmModel=Model
    before=sys.modules.get('transformers');sys.modules['transformers']=fake
    try:
        with tempfile.TemporaryDirectory() as td:
            seq='A'*2044+'M'
            df=__import__('pandas').DataFrame([dict(sequence=seq,reactants_smiles='CCO',products_smiles='CC=O')])
            redox.validate(df)
            v=redox.protein_features([seq],td,'cpu')
            assert calls==[1022,1022,1]
            np.testing.assert_allclose(v,np.full((1,480),(2044+3)/2045),rtol=1e-6)
            again=redox.protein_features([seq],td,'cpu');np.testing.assert_array_equal(v,again)
            assert len(calls)==3
        print('PASS: long sequence fully covered, length-weighted pooling, cache reload; fake ESM only.')
    finally:
        if before is None:sys.modules.pop('transformers',None)
        else:sys.modules['transformers']=before


if __name__=='__main__':main()

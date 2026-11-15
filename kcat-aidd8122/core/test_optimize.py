"""Small actual regressors + chemistry test; protein vectors are synthetic (no ESM download)."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
import redox
import optimize as opt


def main():
    rows = []
    pairs = [('CCO','CC=O'), ('CCCO','CCC=O'), ('CCCCO','CCCC=O')]
    for i in range(90):
        r, p = pairs[i % 3]
        rows.append(dict(sequence='M'+'A'*(i+1), reactants_smiles=r, products_smiles=p,
                         kcat_s_1=10**((i%13)/4-1), split_group=i//3, ec='1.1.1.1'))
    df = pd.DataFrame(rows)
    swapped = df.copy()
    swapped['reactants_smiles'] = df.products_smiles
    swapped['products_smiles'] = df.reactants_smiles
    m = opt.chemistry(df, 'ordered'); ms = opt.chemistry(swapped, 'ordered')
    assert np.array_equal(m[:,:2048], ms[:,2048:]) and not np.array_equal(m, ms)
    assert np.array_equal(opt.chemistry(df,'ordered_diff')[:,4096:], -opt.chemistry(swapped,'ordered_diff')[:,4096:])
    assert np.array_equal(opt.chemistry(df,'drfp'), opt.chemistry(swapped,'drfp'))
    def fake_protein(frame, *args):
        return np.stack([np.random.default_rng(len(s)).normal(size=480).astype(np.float32) for s in frame.sequence])
    original = opt.protein; opt.protein = fake_protein
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); inp = root/'train.csv'; df.to_csv(inp,index=False)
            x = np.concatenate([fake_protein(df),m],axis=1)
            redox.fit_model(df, x, root/'baseline', trees=3, jobs=1)
            (root/'baseline/input_sha256.txt').write_text(opt.digest(inp))
            splits = root/'baseline/splits.csv'
            tr, va, te = opt.split_indices(df,splits,inp)
            a = SimpleNamespace(input=str(inp), splits=str(splits), output=str(root/'search'),
                 cache=str(root/'cache'), device='cpu', protein='35M', jobs=1,
                 representations=['ordered','ordered_diff','drfp'], models=['extra','catboost'],
                 trials=1,trees=5,seeds=[42,43])
            opt.run_search(a)
            assert not (root/'search/test_predictions.csv').exists()
            board = pd.read_csv(root/'search/leaderboard.csv'); assert len(board)==6
            bundle = joblib.load(root/'search/model.joblib')
            val = pd.read_csv(root/'search/validation_predictions.csv')
            xx = opt.feature_matrix(df,a,bundle['config']['representation'])
            pred = np.mean([m.predict(xx[va]) for m in bundle['models']],axis=0)
            np.testing.assert_allclose(pred,val.pred_log10_kcat,rtol=1e-6)
            # Explicit test evaluation and inference produce the same held-out values.
            a.command='evaluate'; a.model=str(root/'search/model.joblib'); a.output=str(root/'evaluation')
            opt.infer(a)
            held = pd.read_csv(root/'evaluation/test_predictions.csv')
            np.testing.assert_allclose(held.pred_log10_kcat,np.mean([m.predict(xx[te]) for m in bundle['models']],axis=0))
            a.command='predict'; a.output=str(root/'predictions.csv'); opt.infer(a)
            # Changed input must not silently reuse the baseline split.
            altered=root/'altered.csv'; df.iloc[::-1].to_csv(altered,index=False)
            try: opt.split_indices(df.iloc[::-1],splits,altered)
            except ValueError: pass
            else: raise AssertionError('accepted altered input')
            import neural
            a.output=str(root/'mlp'); a.epochs=3; a.patience=2; a.batch_size=16; a.seed=42
            neural.train(a)
            fn, config=neural.load_predictor(root/'mlp/model.pt','cpu')
            saved=pd.read_csv(root/'mlp/validation_predictions.csv')
            np.testing.assert_allclose(fn(x[va]),saved.pred_log10_kcat,rtol=1e-5,atol=1e-6)
            import torch
            checkpoint=torch.load(root/'mlp/model.pt',weights_only=True)
            np.testing.assert_allclose(checkpoint['protein_mean'], x[tr,:480].mean(0))
            assert not (root/'mlp/test_predictions.csv').exists()
            a.command='evaluate'; a.model=str(root/'mlp/model.pt'); a.output=str(root/'mlp_eval'); opt.infer(a)
            a.command='predict'; a.output=str(root/'mlp_predict.csv'); opt.infer(a)
        print('PASS: real ExtraTrees/CatBoost/DRFP/CPU-MLP; split hash; train-only scaling; validation selection; save/reload; evaluate/predict. ESM mocked.')
    finally:
        opt.protein=original


if __name__=='__main__': main()

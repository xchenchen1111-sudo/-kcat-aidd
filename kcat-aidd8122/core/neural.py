"""Frozen-feature fusion MLP; train-only normalization and validation early stopping."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import redox
from optimize import split_indices, feature_matrix, digest, write_json, diagnostics


def device_name(name):
    return ('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name


class Fusion(nn.Module):
    def __init__(self, protein_dim):
        super().__init__()
        self.protein_dim = protein_dim
        self.protein = nn.Sequential(nn.Linear(protein_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.molecule = nn.Sequential(nn.Linear(2048, 256), nn.LayerNorm(256), nn.GELU())
        self.reaction = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(.2),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        d = self.protein_dim
        e = self.protein(x[:, :d])
        r = self.molecule(x[:, d:d+2048]); p = self.molecule(x[:, d+2048:])
        q = self.reaction(torch.cat([r, p, p-r], dim=1))
        return self.head(torch.cat([e, q, e*q], dim=1)).squeeze(-1)


def transform(x, state):
    d = state['protein_dim']; out = np.array(x, dtype=np.float32, copy=True)
    out[:, :d] = (out[:, :d] - np.asarray(state['protein_mean'])) / np.asarray(state['protein_std'])
    out[:, d:] = np.log1p(out[:, d:])
    return out


def predict_batches(model, x, state, device):
    model.eval(); values = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            batch = torch.from_numpy(transform(x[start:start+256], state)).to(device)
            values.append(model(batch).cpu().numpy())
    return np.concatenate(values) * state['target_std'] + state['target_mean']


def load_predictor(path, device):
    device = device_name(device)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = Fusion(state['protein_dim']).to(device)
    model.load_state_dict(state['weights'])
    return lambda x: predict_batches(model, x, state, device), state['config']


def train(a):
    if a.epochs < 1 or a.patience < 1 or a.batch_size < 1 or a.jobs < 1:
        raise ValueError('epochs/patience/batch-size/jobs must be positive')
    torch.set_num_threads(a.jobs)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    df = pd.read_csv(a.input); redox.validate(df, True)
    tr, va, _ = split_indices(df, a.splits, a.input)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    x = feature_matrix(df, a, 'ordered'); y = np.log10(df.kcat_s_1.to_numpy(float))
    d = 480 if a.protein == '35M' else 1280
    config = dict(protein=a.protein, representation='ordered', family='fusion_mlp', seed=a.seed,
                  protein_pooling='length_weighted_nonoverlapping_1022_residue_windows',
                  input_sha256=digest(a.input), split_sha256=digest(a.splits),
                  selection='minimum validation RMSE epoch', learning_rate=.001, weight_decay=.0001,
                  epochs=a.epochs, patience=a.patience, batch_size=a.batch_size)
    state = dict(protein_dim=d, protein_mean=x[tr, :d].mean(0).tolist(),
                 protein_std=np.maximum(x[tr, :d].std(0), 1e-6).tolist(),
                 target_mean=float(y[tr].mean()), target_std=max(float(y[tr].std()), 1e-6), config=config)
    xt = transform(x[tr], state)
    yt = ((y[tr] - state['target_mean']) / state['target_std']).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt)),
                        batch_size=a.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    device = device_name(a.device); model = Fusion(d).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, stale, history = float('inf'), 0, []
    for epoch in range(1, a.epochs+1):
        model.train(); total = 0.
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); loss = nn.functional.mse_loss(model(bx), by)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            total += loss.item() * len(bx)
        pred = predict_batches(model, x[va], state, device)
        score = redox.metrics(y[va], pred)
        if not np.isfinite(score['RMSE_log10']): raise RuntimeError('Nonfinite validation loss')
        row = dict(epoch=epoch, train_normalized_mse=total/len(tr), **score)
        history.append(row); pd.DataFrame(history).to_csv(out/'history.csv', index=False)
        print(row, flush=True)
        if score['RMSE_log10'] < best:
            best = score['RMSE_log10']; stale = 0
            state['weights'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state['best_epoch'] = epoch; torch.save(state, out/'model.pt')
        else:
            stale += 1
            if stale >= a.patience: break
    predict, _ = load_predictor(out/'model.pt', device)
    report = {}
    for name, ids in [('train', tr), ('validation', va)]:
        pred = predict(x[ids]); report[name] = redox.metrics(y[ids], pred)
        diagnostics(df, ids, y, pred, out, name)
    write_json(out/'metrics.json', report); write_json(out/'config.json', config)
    print('MLP complete. Test set NOT evaluated.', flush=True)

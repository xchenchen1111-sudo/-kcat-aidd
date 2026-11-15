"""Validation-only model selection. Test evaluation is an explicit separate command."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
import redox


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')


def split_indices(df, path, input_path):
    path = Path(path)
    expected = path.parent / 'input_sha256.txt'
    if not expected.exists() or expected.read_text().strip() != digest(input_path):
        raise ValueError('Split input hash missing/mismatched: use baseline splits with its original train.csv')
    s = pd.read_csv(path, dtype={'split_group': str})
    if len(s) != len(df) or not np.array_equal(s.row.to_numpy(), np.arange(len(df))):
        raise ValueError('Split row order/length mismatch')
    if not np.array_equal(s.split_group.to_numpy(), df.split_group.astype(str).to_numpy()):
        raise ValueError('Split groups mismatch')
    if set(s['split']) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train/validation/test partitions')
    ids = [np.flatnonzero(s['split'].to_numpy() == k) for k in ['train', 'validation', 'test']]
    for i in range(3):
        for j in range(i):
            if set(s.iloc[ids[i]].split_group) & set(s.iloc[ids[j]].split_group):
                raise ValueError('Group leakage')
            if set(df.iloc[ids[i]].sequence) & set(df.iloc[ids[j]].sequence):
                raise ValueError('Sequence leakage')
    return ids


def protein(df, cache, device, size):
    # Same cache namespace as the existing script for the 35M model.
    redox.MODEL = f'hf_esm2_{"t12_35M" if size == "35M" else "t33_650M"}_UR50D_v1'
    redox.HF_MODEL = f'facebook/esm2_{"t12_35M" if size == "35M" else "t33_650M"}_UR50D'
    return redox.protein_features(df.sequence.tolist(), cache, device,
                                  feature_dim=480 if size == '35M' else 1280)


def chemistry(df, representation):
    if representation == 'drfp':
        from drfp import DrfpEncoder
        return np.asarray(DrfpEncoder.encode(
            (df.reactants_smiles + '>>' + df.products_smiles).tolist(),
            n_folded_length=2048, radius=3), dtype=np.float32)
    m = redox.molecular_features(df)
    if representation == 'reactants':
        return m[:, :2048]
    if representation == 'ordered':
        return m
    if representation == 'ordered_diff':
        return np.concatenate([m, m[:, 2048:] - m[:, :2048]], axis=1)
    raise ValueError(representation)


def feature_matrix(df, a, representation):
    return np.concatenate([protein(df, a.cache, a.device, a.protein), chemistry(df, representation)], axis=1)


def candidates(family, count, seed, trees, jobs):
    rng = np.random.default_rng(seed)
    if family == 'extra':
        yield dict(n_estimators=trees, min_samples_leaf=2, max_features=.5, max_depth=None)
        seen = {(2, .5, None)}
        while len(seen) < count:
            key = (int(rng.choice([1, 2, 4, 8])), float(rng.choice([.25, .5, 1.])),
                   rng.choice([0, 16, 32, 64]).item())
            key = (key[0], key[1], key[2] or None)
            if key not in seen:
                seen.add(key)
                yield dict(n_estimators=trees, min_samples_leaf=key[0], max_features=key[1], max_depth=key[2])
    else:
        for _ in range(count):
            yield dict(iterations=trees, depth=int(rng.choice([4, 6, 8])),
                       learning_rate=float(rng.choice([.02, .05, .1])),
                       l2_leaf_reg=float(rng.choice([3, 10, 30])))


def fit_tree(family, params, x, y, tr, va, seed, jobs):
    if family == 'extra':
        model = ExtraTreesRegressor(**params, random_state=seed, n_jobs=jobs)
        model.fit(x[tr], y[tr])
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**params, loss_function='RMSE', random_seed=seed,
                                  thread_count=jobs, allow_writing_files=False, verbose=False)
        model.fit(x[tr], y[tr], eval_set=(x[va], y[va]), early_stopping_rounds=60)
    return model


def diagnostics(df, ids, y, pred, folder, name):
    folder = Path(folder)
    result = df.iloc[ids].copy()
    result.insert(0, 'source_row', ids)
    result['true_log10_kcat'] = y[ids]
    result['pred_log10_kcat'] = pred
    result['absolute_error_log10'] = np.abs(y[ids] - pred)
    result.to_csv(folder / f'{name}_predictions.csv', index=False)
    if 'ec' in result:
        result['ec_subclass'] = result.ec.astype(str).str.split('.').str[:2].str.join('.')
        rows = []
        for ec, part in result.groupby('ec_subclass'):
            rows.append({'ec_subclass': ec, 'n': len(part),
                         **redox.metrics(part.true_log10_kcat, part.pred_log10_kcat)})
        pd.DataFrame(rows).to_csv(folder / f'{name}_by_ec.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y[ids], pred, s=8, alpha=.35)
    low, high = min(y[ids].min(), pred.min()), max(y[ids].max(), pred.max())
    ax[0].plot([low, high], [low, high], 'k--')
    ax[0].set(xlabel='Measured log10(kcat)', ylabel='Predicted log10(kcat)', title=name)
    ax[1].hist(pred - y[ids], bins=35)
    ax[1].set(xlabel='Prediction - measurement (log10)', ylabel='Count')
    fig.tight_layout(); fig.savefig(folder / f'{name}_diagnostics.png', dpi=180); plt.close(fig)


def run_search(a):
    df = pd.read_csv(a.input); redox.validate(df, True)
    tr, va, _ = split_indices(df, a.splits, a.input)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    y = np.log10(df.kcat_s_1.to_numpy(float))
    p = protein(df, a.cache, a.device, a.protein)
    records, best = [], float('inf')
    for rep in a.representations:
        x = np.concatenate([p, chemistry(df, rep)], axis=1)
        for family in a.models:
            for trial, params in enumerate(candidates(family, a.trials, 42, a.trees, a.jobs)):
                scores, models = [], []
                for seed in a.seeds:
                    model = fit_tree(family, params, x, y, tr, va, seed, a.jobs)
                    scores.append(redox.metrics(y[va], model.predict(x[va])))
                    models.append(model)
                prediction = np.mean([m.predict(x[va]) for m in models], axis=0)
                score = redox.metrics(y[va], prediction)
                row = dict(representation=rep, family=family, trial=trial, params=json.dumps(params),
                           seed_rmse_mean=float(np.mean([s['RMSE_log10'] for s in scores])),
                           seed_rmse_std=float(np.std([s['RMSE_log10'] for s in scores])), **score)
                records.append(row)
                pd.DataFrame(records).sort_values('RMSE_log10').to_csv(out/'leaderboard.csv', index=False)
                print(json.dumps(row), flush=True)
                if score['RMSE_log10'] < best:
                    best = score['RMSE_log10']
                    config = dict(protein=a.protein, representation=rep, family=family,
                                  protein_pooling='length_weighted_nonoverlapping_1022_residue_windows',
                                  params=params, seeds=a.seeds, input_sha256=digest(a.input),
                                  split_sha256=digest(a.splits), selection='validation ensemble RMSE only')
                    joblib.dump(dict(models=models, config=config), out/'model.joblib')
                    write_json(out/'config.json', config)
    bundle = joblib.load(out/'model.joblib')
    x = np.concatenate([p, chemistry(df, bundle['config']['representation'])], axis=1)
    report = {}
    for name, ids in [('train', tr), ('validation', va)]:
        pred = np.mean([m.predict(x[ids]) for m in bundle['models']], axis=0)
        report[name] = redox.metrics(y[ids], pred)
        diagnostics(df, ids, y, pred, out, name)
    report['validation_median_baseline'] = redox.metrics(y[va], np.full(len(va), np.median(y[tr])))
    write_json(out/'metrics.json', report)
    versions = {}
    for pkg in ['numpy','pandas','scikit-learn','rdkit','torch','transformers','catboost','drfp']:
        try: versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError: pass
    write_json(out/'versions.json', versions)
    print('Selection complete. Test set NOT evaluated.', flush=True)


def infer(a):
    df = pd.read_csv(a.input)
    model_path = Path(a.model)
    if model_path.suffix == '.pt':
        from neural import load_predictor
        predict, config = load_predictor(model_path, a.device)
    else:
        bundle = joblib.load(model_path); config = bundle['config']
        predict = lambda x: np.mean([m.predict(x) for m in bundle['models']], axis=0)
    redox.validate(df, a.command == 'evaluate')
    a.protein = config['protein']
    if a.command == 'evaluate':
        if digest(a.input) != config['input_sha256'] or digest(a.splits) != config['split_sha256']:
            raise ValueError('Training data/split hash mismatch')
        tr, _, te = split_indices(df, a.splits, a.input)
        x = feature_matrix(df.iloc[te], a, config['representation'])
        pred = predict(x); y = np.log10(df.kcat_s_1.to_numpy(float))
        out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
        write_json(out/'test_metrics.json', dict(test=redox.metrics(y[te], pred),
                   median_baseline=redox.metrics(y[te], np.full(len(te), np.median(y[tr]))),
                   model_sha256=digest(model_path)))
        diagnostics(df, te, y, pred, out, 'test')
    else:
        pred = predict(feature_matrix(df, a, config['representation']))
        df['pred_log10_kcat'] = pred; df['pred_kcat_s_1'] = 10 ** pred
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.output, index=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['search', 'mlp', 'evaluate', 'predict']:
        p = sub.add_parser(name)
        p.add_argument('--input', required=True); p.add_argument('--output', required=True)
        p.add_argument('--cache', default='cache'); p.add_argument('--device', default='cuda', choices=['cpu','cuda','auto'])
        if name != 'predict': p.add_argument('--splits', required=True)
        if name in ['search', 'mlp']:
            p.add_argument('--protein', default='35M', choices=['35M','650M'])
            p.add_argument('--jobs', type=int, default=4)
        else: p.add_argument('--model', required=True)
        if name == 'search':
            p.add_argument('--models', nargs='+', default=['extra','catboost'], choices=['extra','catboost'])
            p.add_argument('--representations', nargs='+', default=['ordered','ordered_diff','drfp'], choices=['reactants','ordered','ordered_diff','drfp'])
            p.add_argument('--trials', type=int, default=8)
            p.add_argument('--trees', type=int, default=500)
            p.add_argument('--seeds', nargs='+', type=int, default=[42])
        if name == 'mlp':
            p.add_argument('--epochs', type=int, default=200); p.add_argument('--patience', type=int, default=20)
            p.add_argument('--batch-size', type=int, default=64); p.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    if a.command == 'search':
        if not 1 <= a.trials <= 48 or a.trees < 1 or a.jobs < 1: parser.error('trials 1..48; trees/jobs positive')
        run_search(a)
    elif a.command == 'mlp':
        from neural import train
        train(a)
    else: infer(a)


if __name__ == '__main__':
    main()

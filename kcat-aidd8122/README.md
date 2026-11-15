# Enzyme turnover prediction with protein and reaction features

AIDD8122 final project — 徐晨晨 (p2625501).

This repository contains the source and compact results for comparing frozen ESM2 embeddings, reaction fingerprints, Fusion networks, ensembles and losses for log10(kcat / s^-1) prediction. No clinical or state-of-the-art claim is made.

## Results

| Held-out model | RMSE | MAE | R2 | Within tenfold |
|---|---:|---:|---:|---:|
| 35M three-seed ensemble | 1.38652 | 1.01749 | -0.0060 | 62.17% |
| 650M three-seed ensemble | 1.32384 | 0.97333 | 0.0829 | 63.07% |
| Fixed equal mixture | 1.33611 | 0.97587 | 0.0658 | 64.06% |

Mixture versus 35M reduces RMSE by 3.64%. Paired bootstrap resamples 75 groups; the 95% interval for RMSE difference is [-0.08519, -0.01790]. Low activity predictions remain poor. The test set has no log10(kcat) >= 4 records.

## Layout

- `core/`: original data preparation, feature extraction, Fusion training and diagnostics, including original tests.
- `experiments/baseline/`: Ridge and small MLP comparisons and verified feature export.
- `experiments/group_cv/`: grouped development validation.
- `experiments/encoder650/`: 650M export and grouped comparison.
- `experiments/loss_study/`: post-test exploratory loss comparisons.
- `experiments/final/`: fixed final training and held-out evaluation.
- `results/`: small aggregate result tables; final test values transcribed from server output.

## Environment

Run commands from the repository root on Linux. Create an isolated Python environment and install `requirements.txt`. Install a PyTorch build compatible with your GPU separately if needed. Dependency versions are not locked: the exact server environment has not yet been exported. Source syntax is checked locally; full training was performed on the server before repository packaging. GPU indices should be adapted to your machine.

## Data

The experiment uses MCKcat-DB.csv and the supplied N20test.csv. Obtain the original release associated with https://doi.org/10.1002/advs.77789 and retain its source manifest. These data and pretrained weights are NOT redistributed here. Confirm the release terms and exact files before use. Put the CSVs under `core/raw/`. Exact reproduction requires the original prepared input/split hashes; data release changes may change the results.

```bash
python core/prepare_data.py --raw core/raw/MCKcat-DB.csv --author-test core/raw/N20test.csv --output core/data
python core/optimize.py mlp --input core/data/train.csv --splits core/data/splits.csv --output core/runs/mlp35_seed42 --cache cache --protein 35M --seed 42 --device cuda
mkdir -p features
python experiments/baseline/export_features.py --project core --run core/runs/mlp35_seed42 --cache cache --output features/features35.npz
```

Feature export checks reproduction of original predictions. ESM model weights may be downloaded by the core feature extractor. For a 650M workflow, prepopulate the official `facebook/esm2_t33_650M_UR50D` model cache: the export/final scripts intentionally set Hugging Face offline mode.

```bash
python experiments/group_cv/run_cv.py --features features/features35.npz --train-csv core/runs/mlp35_seed42/train_predictions.csv --out group35_results --device cuda:0
CUDA_VISIBLE_DEVICES=0 python experiments/encoder650/export_650m.py --project core --cache cache --source features/features35.npz --train-csv core/runs/mlp35_seed42/train_predictions.csv --out features/features650.npz
python experiments/encoder650/run_cv.py --features features/features650.npz --train-csv core/runs/mlp35_seed42/train_predictions.csv --out group650_results --device cuda:0
```

The feature export requires at least 8 GiB free on the destination filesystem. Final evaluation requires a fresh output folder and the reference run config for input/split hash verification:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/final/final_evaluate.py --project core --cache cache --out final_results
```

It trains 35M and 650M with seeds 42/43/44, uses original validation for stopping, then evaluates the test once. Native final checkpoints use `core/neural.py`; experimental checkpoints use their own `train_fusion.py` and normalization archive. These checkpoint formats are not interchangeable. Use the dedicated final script for held-out evaluation, not the generic original evaluate command on its development-only checkpoint config.

## Evaluation boundaries

Prepared partitions: train 24,480; validation 2,759; test 2,440. Groups use accession/exact sequence, NOT independently verified homology clusters. The source preprocessing script also notes that previous EC1 experiments share source data, so this is not a fully independent external dataset. Development folds are also used for early stopping; scores are not unbiased outer nested-CV estimates. Huber and weighted-loss comparisons were performed after the final test was inspected and are exploratory only. Do not retune on the reported test labels.

## References and attribution

- Li et al. (2022), DLKcat: https://doi.org/10.1038/s41929-022-00798-z
- Kroll et al. (2023), TurNuP: https://doi.org/10.1038/s41467-023-39840-4
- Lin et al. (2023), ESM: https://pubmed.ncbi.nlm.nih.gov/36927031/
- Official pretrained encoders: https://github.com/facebookresearch/esm
- MCKcat study: https://doi.org/10.1002/advs.77789

Code was developed and revised with AI assistance and executed by the student on the research server. Verify and disclose assistance according to course policy. No third-party software or dataset license is replaced by this repository. An open-source license has not been assigned; confirm ownership and reuse rights before adding one.

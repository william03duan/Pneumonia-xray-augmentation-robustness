# Quality-Aware Augmentation for Pneumonia Classification

This project examined whether moderate, label-preserving image-quality augmentation improved pneumonia classification robustness to degraded chest X-rays without harming performance on clean images. It compared a compact residual CNN and a compact Vision Transformer (ViT) on PneumoniaMNIST.

## Repository layout

```text
src/            CNN, ViT, and result-figure source code
data/           Local PneumoniaMNIST NPZ file (not tracked)
data_preview/   GitHub-viewable dataset summary and sample X-rays
results/runs/   Metrics, predictions, and figures from the four experiments
report/         NeurIPS-style LaTeX report, bibliography, style file, and figures
lectures/       Lectrace presentation source and its assets
scripts/        Dataset-preview generation utility
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the 128 by 128 PneumoniaMNIST archive and save it as `data/pneumoniamnist_128.npz`; see [data/README.md](data/README.md).

## Reproduce the four experiments

```bash
python src/cnn_model.py --run-name cnn_baseline
python src/cnn_model.py --quality-augmentation --run-name cnn_augmented
python src/transformer_model.py --run-name transformer_baseline
python src/transformer_model.py --quality-augmentation --run-name transformer_augmented
```

Each command writes a timestamped directory under `results/runs/`. The project retains metrics, predictions, and plots in Git; large `best_model.pt` checkpoints remain local by default.

## Recreate the main robustness figure

```bash
python src/create_results_figure.py \
  --c0 results/runs/cnn_baseline_20261025-084512/metrics.json \
  --c1 results/runs/cnn_augmented_20261025-214602/metrics.json \
  --t0 results/runs/transformer_baseline_20261025-090900/metrics.json \
  --t1 results/runs/transformer_augmented_20261025-215518/metrics.json \
  --output report/figures/robustness_auroc_comparison.png
```

## Lectrace presentation

```bash
pip install -r lectures/requirements.txt
lectrace serve --port 7001 lectures/01_pneumonia_robustness.py
```

Then open `http://localhost:7001` while the terminal process is running.

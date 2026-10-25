# Quality-Aware Augmentation for Robust Pneumonia Classification

## Overview
Can quality-aware data augmentation improve pneumonia classification robustness on degraded chest X-rays without reducing performance on clean images?
This project trained a compact residual CNN and a compact Vision Transformer (ViT) on PneumoniaMNIST. Each architecture was trained as a clean-only control and with moderate, label-preserving quality augmentation: Gaussian blur, Gaussian noise, contrast reduction, or reduced resolution.

## Main result

Quality-aware augmentation improved the CNN's mean AUROC across seven degraded test conditions from **0.907** to **0.948**, while clean-test AUROC increased from **0.955** to **0.962**. For the ViT, clean AUROC changed from **0.953** to **0.951** and mean degraded AUROC remained **0.934**. In this single-seed benchmark, the augmentation benefit was therefore architecture-dependent.

### Clean-test performance

| Metric | C0: CNN control | C1: CNN augmented | T0: ViT control | T1: ViT augmented |
| --- | ---: | ---: | ---: | ---: |
| AUROC | 0.955 | **0.962** | **0.953** | 0.951 |
| AUPRC | 0.967 | **0.974** | **0.969** | 0.968 |
| Balanced accuracy | 0.813 | 0.821 | **0.867** | 0.839 |
| Macro-F1 | 0.833 | 0.842 | **0.877** | 0.856 |
| Sensitivity | 0.990 | **0.997** | 0.951 | 0.977 |
| Specificity | 0.637 | 0.645 | **0.782** | 0.701 |

### Robustness under controlled quality degradation

AUROC was evaluated on the same held-out test images after deterministic corruptions. ``Mean degraded'' is the average over the seven degraded conditions, excluding clean images.

| Test condition | C0: CNN control | C1: CNN augmented | T0: ViT control | T1: ViT augmented |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.955 | **0.962** | **0.953** | 0.951 |
| Blur, $\sigma=1.0$ | 0.951 | **0.959** | **0.955** | 0.953 |
| Blur, $\sigma=2.0$ | 0.927 | **0.949** | **0.956** | 0.954 |
| Noise, SD = 0.03 | 0.934 | **0.954** | 0.954 | 0.954 |
| Noise, SD = 0.07 | 0.878 | **0.934** | 0.940 | **0.948** |
| Contrast, 0.60 | 0.932 | **0.952** | **0.941** | 0.923 |
| Contrast, 0.35 | 0.781 | **0.932** | 0.834 | **0.851** |
| Low resolution, 0.50 | 0.950 | **0.958** | **0.956** | 0.953 |
| **Mean degraded** | 0.907 | **0.948** | **0.934** | **0.934** |

![AUROC across clean and degraded test conditions](assets/robustness_auroc_comparison.png)

## Experimental design

| ID | Architecture | Training policy |
| --- | --- | --- |
| C0 | CNN | Clean-only control |
| C1 | CNN | Quality-aware augmentation |
| T0 | Vision Transformer | Clean-only control |
| T1 | Vision Transformer | Quality-aware augmentation |

The held-out test images were evaluated clean and under blur, noise, contrast, and low-resolution corruptions. AUROC was the primary metric.

## Repository layout

```text
src/            CNN, ViT, and result-figure source code
data/           Local PneumoniaMNIST NPZ archive (not tracked)
data_preview/   GitHub-viewable class summary and sample chest X-rays
assets/         Figures displayed in this README
results/runs/   Metrics, predictions, and figures from four completed runs
lectures/       Lectrace presentation source and assets
scripts/        Dataset-preview generation utility
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `pneumoniamnist_128.npz` from the official [MedMNIST v2 project](https://medmnist.com/) and save it as `data/pneumoniamnist_128.npz`. The source archive is intentionally excluded from Git; see [data/README.md](data/README.md).

## Run an experiment

```bash
# CNN: clean-only control and quality-augmented treatment
python src/cnn_model.py --run-name cnn_baseline
python src/cnn_model.py --quality-augmentation --run-name cnn_augmented

# ViT: clean-only control and quality-augmented treatment
python src/transformer_model.py --run-name transformer_baseline
python src/transformer_model.py --quality-augmentation --run-name transformer_augmented
```

Each run writes its metrics, predictions, and plots to `results/runs/`. Large model checkpoints are kept locally and excluded from Git.

## Lectrace presentation

```bash
pip install -r lectures/requirements.txt
lectrace serve --port 7001 lectures/01_pneumonia_robustness.py
```

Open `http://localhost:7001` while the process is running.

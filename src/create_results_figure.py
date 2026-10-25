"""Create the robustness figure used in pneumonia_robustness_report.tex.

Example:
python create_results_figure.py \
  --c0 /path/to/cnn_baseline/metrics.json \
  --c1 /path/to/cnn_augmented/metrics.json \
  --t0 /path/to/transformer_baseline/metrics.json \
  --t1 /path/to/transformer_augmented/metrics.json \
  --output figures/robustness_auroc_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = (
    ("clean", "Clean"),
    ("blur_s1.0", "Blur\n$\\sigma=1.0$"),
    ("blur_s2.0", "Blur\n$\\sigma=2.0$"),
    ("noise_s0.03", "Noise\n$\\sigma=0.03$"),
    ("noise_s0.07", "Noise\n$\\sigma=0.07$"),
    ("contrast_0.60", "Contrast\n0.60"),
    ("contrast_0.35", "Contrast\n0.35"),
    ("lowres_0.50", "Low resolution\n0.50"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("c0", "c1", "t0", "t1"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aurocs(metrics_path: Path) -> list[float]:
    with metrics_path.open(encoding="utf-8") as file:
        conditions = json.load(file)["test_conditions"]
    return [conditions[key]["auroc"] for key, _ in CONDITIONS]


def plot_pair(axis: plt.Axes, control: list[float], augmented: list[float], title: str) -> None:
    x = list(range(len(CONDITIONS)))
    axis.plot(x, control, marker="o", linewidth=2, label="Clean-only control")
    axis.plot(x, augmented, marker="o", linewidth=2, label="Quality-augmented")
    axis.set_xticks(x, [label for _, label in CONDITIONS], fontsize=8)
    axis.set_ylim(0.75, 0.98)
    axis.grid(axis="y", alpha=0.3)
    axis.set_title(title)
    axis.set_ylabel("Test AUROC")


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    plot_pair(axes[0], aurocs(args.c0), aurocs(args.c1), "CNN")
    plot_pair(axes[1], aurocs(args.t0), aurocs(args.t1), "Vision Transformer")
    axes[1].legend(loc="lower left", fontsize=9)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CNN model for PneumoniaMNIST.

Run from the repository root:
    python src/cnn_model.py --quality-augmentation --run-name cnn_augmented
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = PROJECT_ROOT / ".matplotlib"

MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib


matplotlib.use("Agg")  # Must precede pyplot: this script saves PNG files without a GUI.
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, \
    precision_recall_curve, recall_score, roc_auc_score, roc_curve
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA = PROJECT_ROOT / "data" / "pneumoniamnist_128.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "runs"
CORRUPTIONS = ("clean", "blur_s1.0", "blur_s2.0", "noise_s0.03", "noise_s0.07", "contrast_0.60",
               "contrast_0.35", "lowres_0.50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone CNN for robust pneumonia classification.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"Dataset NPZ (default: {DEFAULT_DATA})")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quality-augmentation", action="store_true")
    parser.add_argument("--keep-validation-duplicates", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_split(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        images = data[f"{split}_images"].copy()
        labels = data[f"{split}_labels"].reshape(-1).astype(np.int64, copy=True)
    if images.ndim != 3 or images.dtype != np.uint8 or len(images) != len(labels):
        raise ValueError("Expected uint8 image arrays shaped [N, H, W] and binary labels.")
    return images, labels


def remove_validation_duplicates(train: np.ndarray, val: np.ndarray, val_labels: np.ndarray) -> \
tuple[np.ndarray, np.ndarray, int]:
    train_hashes = {hashlib.sha256(image.tobytes()).digest() for image in train}
    keep = np.array([hashlib.sha256(image.tobytes()).digest() not in train_hashes for image in val])
    return val[keep], val_labels[keep], int((~keep).sum())


def mean_std(images: np.ndarray) -> tuple[float, float]:
    total, squares, n = 0, 0.0, images.size
    for start in range(0, len(images), 256):
        chunk = images[start: start + 256]
        total += int(chunk.sum(dtype=np.uint64));
        squares += float(np.square(chunk, dtype=np.float64).sum())
    mean_255 = total / n
    return mean_255 / 255.0, math.sqrt(max(squares / n - mean_255 ** 2, 1e-12)) / 255.0


def blur(image: Tensor, sigma: float) -> Tensor:
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2));
    k = k / k.sum();
    kernel = torch.outer(k, k)[None, None]
    return F.conv2d(F.pad(image[None], (radius,) * 4, mode="reflect"), kernel)[0]


def lower_contrast(image: Tensor, factor: float) -> Tensor:
    m = image.mean(dim=(-2, -1), keepdim=True)
    return ((image - m) * factor + m).clamp(0, 1)


def down_up(image: Tensor, scale: float) -> Tensor:
    h, w = image.shape[-2:]
    x = F.interpolate(image[None], size=(max(8, round(h * scale)), max(8, round(w * scale))),
                      mode="bilinear", align_corners=False)
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)[0]


def augment(image: Tensor) -> Tensor:
    """One moderate quality perturbation; deliberately no horizontal flipping."""
    if torch.rand(()) > .75:
        return image
    which = int(torch.randint(0, 4, ()).item())
    if which == 0:
        return blur(image, float(torch.empty(()).uniform_(.35, 1.25)))
    if which == 1:
        return (image + torch.randn_like(image) * float(torch.empty(()).uniform_(.01, .05))).clamp(
            0, 1)
    if which == 2:
        return lower_contrast(image, float(torch.empty(()).uniform_(.60, 1.0)))
    return down_up(image, float(torch.empty(()).uniform_(.50, .85)))


def corrupt(image: Tensor, condition: str, index: int, seed: int) -> Tensor:
    if condition == "clean":
        return image
    if condition.startswith("blur_s"):
        return blur(image, float(condition.removeprefix("blur_s")))
    if condition.startswith("noise_s"):
        noise = torch.randn(image.shape,
                            generator=torch.Generator().manual_seed(seed * 1_000_003 + index),
                            dtype=image.dtype)
        return (image + noise * float(condition.removeprefix("noise_s"))).clamp(0, 1)
    if condition.startswith("contrast_"):
        return lower_contrast(image, float(condition.removeprefix("contrast_")))
    if condition.startswith("lowres_"):
        return down_up(image, float(condition.removeprefix("lowres_")))
    raise ValueError(condition)


class XRayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, mean: float, std: float,
                 train: bool = False, condition: str = "clean", seed: int = 17) -> None:
        self.images, self.labels, self.mean, self.std = images, labels, mean, std
        self.train, self.condition, self.seed = train, condition, seed

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = torch.from_numpy(self.images[index]).float().div(255).unsqueeze(0)
        if self.train:
            image = augment(image)
        image = corrupt(image, self.condition, index, self.seed)
        return (image - self.mean) / self.std, torch.tensor(self.labels[index], dtype=torch.long)


class Block(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels))
        self.skip: nn.Module = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


class PneumoniaCNN(nn.Module):
    """A compact ResNet-18-style CNN for single-channel chest X-rays."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layers = nn.Sequential(Block(32, 32), Block(32, 32), Block(32, 64, 2), Block(64, 64),
                                    Block(64, 128, 2), Block(128, 128), Block(128, 256, 2),
                                    Block(256, 256))
        self.classifier = nn.Linear(256, 2)

    def forward(self, image: Tensor) -> Tensor:
        return self.classifier(F.adaptive_avg_pool2d(self.layers(self.stem(image)), 1).flatten(1))


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool,
                gpu: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=gpu.type == "cuda")


@torch.inference_mode()
def probabilities(model: nn.Module, batches: DataLoader, gpu: torch.device) -> tuple[
    np.ndarray, np.ndarray]:
    model.eval();
    y, p = [], []
    for image, label in batches:
        p.append(model(image.to(gpu)).softmax(1)[:, 1].cpu().numpy());
        y.append(label.numpy())
    return np.concatenate(y), np.concatenate(p)


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    prediction = (p >= .5).astype(int)
    return {"auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
            "sensitivity": float(recall_score(y, prediction, pos_label=1)),
            "specificity": float(recall_score(y, prediction, pos_label=0))}


def figures(history: list[dict[str, float]], predictions: dict[str, tuple[np.ndarray, np.ndarray]],
            results: dict[str, dict[str, float]], folder: Path) -> None:
    colors = plt.cm.viridis(np.linspace(.08, .90, len(predictions)))
    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    for color, (name, (y, p)) in zip(colors, predictions.items()):
        fpr, tpr, _ = roc_curve(y, p);
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={results[name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance");
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate (sensitivity)",
           title="CNN ROC curves", xlim=(0, 1), ylim=(0, 1));
    ax.grid(alpha=.25);
    ax.legend(fontsize=8, loc="lower right");
    fig.savefig(folder / "roc_curves.png", dpi=220);
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    for color, (name, (y, p)) in zip(colors, predictions.items()):
        precision, recall, _ = precision_recall_curve(y, p);
        ax.plot(recall, precision, color=color,
                label=f"{name} (AP={results[name]['average_precision']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="CNN precision–recall curves", xlim=(0, 1),
           ylim=(0, 1));
    ax.grid(alpha=.25);
    ax.legend(fontsize=8);
    fig.savefig(folder / "precision_recall_curves.png", dpi=220);
    plt.close(fig)
    epoch = [row["epoch"] for row in history];
    fig, left = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    left.plot(epoch, [row["loss"] for row in history], "o-", label="Training loss");
    left.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="CNN training history");
    left.grid(alpha=.25)
    right = left.twinx();
    right.plot(epoch, [row["val_auroc"] for row in history], "s-", color="tab:orange",
               label="Validation AUC");
    right.set(ylabel="Validation AUC", ylim=(0, 1));
    lines = left.lines + right.lines;
    left.legend(lines, [line.get_label() for line in lines]);
    fig.savefig(folder / "training_history.png", dpi=220);
    plt.close(fig)
    names = list(results);
    x = np.arange(len(names));
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(x, [results[n]["auroc"] for n in names], "o-", label="AUROC");
    ax.plot(x, [results[n]["balanced_accuracy"] for n in names], "s-", label="Balanced accuracy");
    ax.set(xticks=x, xticklabels=names, xlabel="Test condition", ylabel="Score", ylim=(0, 1),
           title="CNN robustness");
    ax.tick_params(axis="x", rotation=30);
    ax.grid(axis="y", alpha=.25);
    ax.legend();
    fig.savefig(folder / "robustness_scores.png", dpi=220);
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    seed_everything(args.seed);
    gpu = device()
    train_x, train_y = read_split(args.data, "train");
    val_x, val_y = read_split(args.data, "val");
    test_x, test_y = read_split(args.data, "test")
    removed = 0
    if not args.keep_validation_duplicates:
        val_x, val_y, removed = remove_validation_duplicates(train_x, val_x, val_y)
    mean, std = mean_std(train_x);
    model = PneumoniaCNN().to(gpu)
    train_data = XRayDataset(train_x, train_y, mean, std, train=args.quality_augmentation);
    val_data = XRayDataset(val_x, val_y, mean, std)
    weights = len(train_y) / (2 * np.bincount(train_y, minlength=2));
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=gpu))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay);
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs);
    scaler = torch.cuda.amp.GradScaler(enabled=gpu.type == "cuda")
    run_name = args.run_name or f"cnn_{'augmented' if args.quality_augmentation else 'baseline'}_{time.strftime('%Y%m%d-%H%M%S')}";
    folder = args.output_dir / run_name;
    folder.mkdir(parents=True, exist_ok=False)
    best, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train();
        total, count = 0.0, 0
        for image, label in make_loader(train_data, args, True, gpu):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=gpu.type == "cuda"):
                loss = loss_fn(model(image.to(gpu)), label.to(gpu))
            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update();
            total += float(loss.detach()) * len(label);
            count += len(label)
        scheduler.step();
        vy, vp = probabilities(model, make_loader(val_data, args, False, gpu), gpu);
        score = metric_values(vy, vp)["auroc"];
        history.append({"epoch": epoch, "loss": total / count, "val_auroc": score});
        print(f"Epoch {epoch:02d}: loss={total / count:.4f}; val AUC={score:.4f}")
        if score > best:
            best, best_epoch, stale, best_state = score, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None;
    model.load_state_dict(best_state);
    torch.save({"architecture": "PneumoniaCNN", "state_dict": model.state_dict(), "mean": mean,
                "std": std}, folder / "best_model.pt")
    results, preds, archive = {}, {}, {"labels": test_y}
    for condition in CORRUPTIONS:
        y, p = probabilities(model, make_loader(
            XRayDataset(test_x, test_y, mean, std, condition=condition, seed=args.seed), args,
            False, gpu), gpu);
        results[condition] = metric_values(y, p);
        preds[condition] = (y, p);
        archive["probability_" + condition.replace(".", "_")] = p;
        print(f"{condition:>14}: AUROC={results[condition]['auroc']:.4f}")
    np.savez_compressed(folder / "test_predictions.npz", **archive)
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "loss",
                                                  "val_auroc")); writer.writeheader(); writer.writerows(
            history)
    figures(history, preds, results, folder)
    with (folder / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"model": "PneumoniaCNN", "best_validation_epoch": best_epoch,
                   "best_validation_auroc": best, "validation_duplicates_removed": removed,
                   "test_conditions": results, "data": str(args.data)}, file, indent=2)
    print(f"Saved results to {folder.resolve()}")


if __name__ == "__main__":
    main()

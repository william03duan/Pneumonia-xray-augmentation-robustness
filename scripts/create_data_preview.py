"""Create small, GitHub-readable previews from the local PneumoniaMNIST NPZ.

Run from the repository root:
    .venv/bin/python scripts/create_data_preview.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "pneumoniamnist_128.npz"
OUTPUT = ROOT / "data_preview"
LABELS = {0: "Normal", 1: "Pneumonia"}


def write_summary(data: np.lib.npyio.NpzFile) -> None:
    rows = []
    for split in ("train", "val", "test"):
        labels = data[f"{split}_labels"].reshape(-1)
        for label, label_name in LABELS.items():
            count = int((labels == label).sum())
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "label_name": label_name,
                    "count": count,
                    "proportion": f"{count / len(labels):.4f}",
                }
            )
    with (OUTPUT / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_sample_montage(data: np.lib.npyio.NpzFile) -> None:
    images = data["train_images"]
    labels = data["train_labels"].reshape(-1)
    rng = np.random.default_rng(17)
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        selected.extend((int(index), label) for index in rng.choice(indices, size=8, replace=False))

    cell, label_height, columns = 192, 28, 4
    rows = len(selected) // columns
    canvas = Image.new("RGB", (columns * cell, 34 + rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), "PneumoniaMNIST: 8 Normal and 8 Pneumonia training examples", fill="black", font=font)
    for position, (index, label) in enumerate(selected):
        row, column = divmod(position, columns)
        x, y = column * cell, 34 + row * (cell + label_height)
        xray = Image.fromarray(images[index]).resize((cell, cell), Image.Resampling.LANCZOS).convert("RGB")
        canvas.paste(xray, (x, y))
        draw.text((x + 5, y + cell + 6), f"{LABELS[label]} (train)", fill="black", font=font)
    canvas.save(OUTPUT / "sample_xrays.png", optimize=True)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUTPUT.mkdir(exist_ok=True)
    with np.load(DATA_PATH) as data:
        write_summary(data)
        write_sample_montage(data)
    print(f"Wrote GitHub-readable preview to {OUTPUT}")


if __name__ == "__main__":
    main()

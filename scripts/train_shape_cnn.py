#!/usr/bin/env python3
"""Retrain the ShapesCNN feature extractor (optional).

The packaged checkpoint ``checkpoints/cnn-shapes_feat64_100px.pt`` was
produced by an earlier version of this pipeline in 2022; retraining is only
needed if you want to change the encoder.

Data: the 2D geometric shapes dataset (El Korchi & Ghanou, 2020,
https://doi.org/10.17632/wzr2yv7r53.1) — 9 shape classes x 10k RGB images.
``data/shape_dataset/shape_dataset.zip`` (git LFS) holds the 200x200 originals
under ``output/``. This script is a cleaned-up derivative of a classroom
implementation adapted for this project in 2022.

Usage:
    uv run python scripts/train_shape_cnn.py \
        --zip data/shape_dataset/shape_dataset.zip --epochs 30
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from illusion_rnn.models import ShapesCNN  # noqa: E402

# Frozen label map — the shipped checkpoint was trained with these indices.
CLASS_IDS = {
    "Circle": 0, "Square": 1, "Octagon": 2, "Heptagon": 3, "Nonagon": 4,
    "Star": 5, "Hexagon": 6, "Pentagon": 7, "Triangle": 8,
}


class ShapeImages(Dataset):
    def __init__(self, image_dir: Path, img_size: int = 100):
        self.paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if not self.paths:
            msg = f"No images found in {image_dir}"
            raise FileNotFoundError(msg)
        self.img_size = img_size
        self.labels = []
        for path in self.paths:
            label = next(
                (idx for name, idx in CLASS_IDS.items() if name in path.name), None,
            )
            if label is None:
                msg = f"Cannot infer class from filename {path.name!r}"
                raise ValueError(msg)
            self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("L")
        img = img.resize((self.img_size, self.img_size))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        return x.unsqueeze(0), self.labels[index]


def accuracy(loader, model, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path,
                        default=Path("data/shape_dataset/shape_dataset.zip"))
    parser.add_argument("--workdir", type=Path,
                        default=Path("data/shape_dataset/extracted"))
    parser.add_argument("--out", type=Path,
                        default=Path("checkpoints/cnn-shapes_feat64_100px.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    image_dir = args.workdir / "output"
    if not image_dir.exists():
        if args.zip.stat().st_size < 1024:
            msg = (
                f"{args.zip} is an LFS pointer; run "
                "'git lfs install --local && git lfs checkout' first"
            )
            raise SystemExit(msg)
        print(f"Extracting {args.zip} -> {args.workdir}")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(args.workdir)

    dataset = ShapeImages(image_dir)
    n_test = len(dataset) // 5
    train_set, test_set = random_split(
        dataset, [len(dataset) - n_test, n_test],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = ShapesCNN().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        losses = []
        for x, y in train_loader:
            x, y = x.to(args.device), torch.as_tensor(y).to(args.device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(
            f"epoch {epoch}/{args.epochs}  loss {np.mean(losses):.4f}  "
            f"test acc {accuracy(test_loader, model, args.device):.3f}",
        )

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

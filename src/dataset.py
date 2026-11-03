from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .utils import load_json, save_json


def create_split(root: str | Path, split_path: str | Path, seed: int = 42) -> dict:
    root = Path(root)
    names = sorted(p.stem for p in (root / "PNGImages").glob("*.png"))
    if len(names) != 170:
        raise RuntimeError(f"Expected 170 images, found {len(names)} in {root}")
    rng = random.Random(seed)
    rng.shuffle(names)
    split = {"seed": seed, "train": names[:119], "val": names[119:145], "test": names[145:]}
    save_json(split, split_path)
    return split


class PennFudanSemanticDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        split_file = Path(split_file)
        spec = load_json(split_file) if split_file.exists() else create_split(root, split_file, seed)
        self.names = spec[split]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = Image.open(self.root / "PNGImages" / f"{name}.png").convert("RGB")
        mask = Image.open(self.root / "PedMasks" / f"{name}_mask.png").convert("L")
        mask = mask.point(lambda p: 255 if p > 0 else 0)

        if self.augment:
            rng = random.Random(self.seed * 100003 + index + random.randint(0, 10**6))
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
            if rng.random() < 0.35:
                angle = rng.uniform(-7.0, 7.0)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_np = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)[None, ...]
        image_t = torch.from_numpy(image_np)
        image_t = (image_t - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        return image_t, torch.from_numpy(mask_np), name


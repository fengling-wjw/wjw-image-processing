from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import PennFudanSemanticDataset
from src.metrics import segmentation_metrics
from src.models import build_model
from src.utils import save_json, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "improved"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/raw/PennFudanPed")
    p.add_argument("--split-file", default="data/split_seed42.json")
    p.add_argument("--output", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PennFudanSemanticDataset(args.data_root, args.split_file, "test", args.image_size, False, args.seed)
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    model = build_model(args.variant, args.base_channels).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    records, inference = [], []
    with torch.inference_mode():
        for images, masks, names in loader:
            images, masks = images.to(device), masks.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            inference.extend([elapsed * 1000 / images.size(0)] * images.size(0))
            for i, name in enumerate(names):
                item = segmentation_metrics(outputs["mask"][i:i+1], masks[i:i+1])
                records.append({"name": name, **item})
    keys = ["iou", "dice", "precision", "recall", "pixel_accuracy", "boundary_f1"]
    summary = {k: sum(r[k] for r in records) / len(records) for k in keys}
    summary.update({"variant": args.variant, "test_images": len(records), "mean_inference_ms": sum(inference) / len(inference), "checkpoint_epoch": state["epoch"]})
    save_json({"summary": summary, "per_image": records}, args.output)
    print(summary)


if __name__ == "__main__":
    main()


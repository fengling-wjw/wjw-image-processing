from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import PennFudanSemanticDataset
from src.losses import compute_loss
from src.metrics import segmentation_metrics
from src.models import build_model
from src.utils import count_parameters, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline or improved pedestrian segmentation model")
    p.add_argument("--variant", choices=["baseline", "improved"], required=True)
    p.add_argument("--data-root", default="data/raw/PennFudanPed")
    p.add_argument("--split-file", default="data/split_seed42.json")
    p.add_argument("--output", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def make_loaders(args):
    common = dict(root=args.data_root, split_file=args.split_file, image_size=args.image_size, seed=args.seed)
    train_ds = PennFudanSemanticDataset(split="train", augment=True, **common)
    val_ds = PennFudanSemanticDataset(split="val", augment=False, **common)
    return (
        DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True),
        DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True),
    )


@torch.inference_mode()
def validate(model, loader, device, variant):
    model.eval()
    totals = {k: 0.0 for k in ["loss", "iou", "dice", "precision", "recall", "pixel_accuracy", "boundary_f1"]}
    samples = 0
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss, _ = compute_loss(outputs, masks, variant)
        metrics = segmentation_metrics(outputs["mask"], masks)
        n = images.size(0)
        totals["loss"] += float(loss) * n
        for key, value in metrics.items():
            totals[key] += value * n
        samples += n
    return {k: v / samples for k, v in totals.items()}


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.output or f"runs/{args.variant}")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out / "config.json")
    train_loader, val_loader = make_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.variant, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    rows, best_iou, stale = [], -1.0, 0
    started = time.time()
    print(f"variant={args.variant} device={device} parameters={count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss, _ = compute_loss(outputs, masks, args.variant)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        metrics = validate(model, val_loader, device, args.variant)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": running / seen, **{f"val_{k}": v for k, v in metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        print(f"epoch={epoch:02d} train={row['train_loss']:.4f} val_iou={metrics['iou']:.4f} val_dice={metrics['dice']:.4f} val_bf1={metrics['boundary_f1']:.4f}")
        if metrics["iou"] > best_iou:
            best_iou, stale = metrics["iou"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "args": vars(args)}, out / "checkpoints" / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    with (out / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"variant": args.variant, "device": str(device), "parameters": count_parameters(model), "best_val_iou": best_iou, "epochs_ran": len(rows), "training_seconds": time.time() - started}
    save_json(summary, out / "train_summary.json")
    print(summary)


if __name__ == "__main__":
    main()


from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import PennFudanSemanticDataset
from src.models import build_model


OUT = ROOT / "results"
COLORS = {"baseline": (49, 97, 164), "improved": (218, 92, 50)}


def font(size=22, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def plot_history():
    series = {}
    for name in ("baseline", "improved"):
        with (ROOT / "runs" / name / "history.csv").open(encoding="utf-8-sig") as f:
            series[name] = list(csv.DictReader(f))
    canvas = Image.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [(60, 90, 680, 550, "val_iou", "Validation IoU"), (750, 90, 1370, 550, "val_boundary_f1", "Boundary F1")]
    for x0, y0, x1, y1, key, title in panels:
        draw.rectangle((x0, y0, x1, y1), outline=(80, 80, 80), width=2)
        draw.text((x0, 35), title, fill="black", font=font(26, True))
        for tick in range(6):
            y = y1 - tick * (y1 - y0) / 5
            value = tick / 5
            draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
            draw.text((x0 - 52, y - 10), f"{value:.1f}", fill=(70, 70, 70), font=font(16))
        max_epoch = max(len(v) for v in series.values())
        for name, rows in series.items():
            points = []
            for row in rows:
                epoch = int(row["epoch"])
                value = float(row[key])
                x = x0 + (epoch - 1) / max(1, max_epoch - 1) * (x1 - x0)
                y = y1 - value * (y1 - y0)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=COLORS[name], width=4)
        draw.text((x0, y1 + 18), "Epoch", fill="black", font=font(18))
    draw.line((1000, 45, 1045, 45), fill=COLORS["baseline"], width=5)
    draw.text((1055, 31), "Baseline U-Net", fill="black", font=font(18))
    draw.line((1180, 45, 1225, 45), fill=COLORS["improved"], width=5)
    draw.text((1235, 31), "Improved", fill="black", font=font(18))
    OUT.mkdir(exist_ok=True)
    canvas.save(OUT / "training_curves.png")


def inverse_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    array = ((tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def overlay(image, mask, color, alpha=0.45):
    base = np.asarray(image).astype(np.float32)
    m = np.asarray(mask).astype(bool)
    tint = np.zeros_like(base)
    tint[:] = color
    base[m] = base[m] * (1 - alpha) + tint[m] * alpha
    return Image.fromarray(base.astype(np.uint8))


def predict_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for variant in ("baseline", "improved"):
        model = build_model(variant).to(device)
        state = torch.load(ROOT / "runs" / variant / "checkpoints" / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        models[variant] = model
    ds = PennFudanSemanticDataset(ROOT / "data/raw/PennFudanPed", ROOT / "data/split_seed42.json", "test", 128, False, 42)
    metric_data = json.loads((ROOT / "runs/improved/test_metrics.json").read_text(encoding="utf-8"))["per_image"]
    ranked = sorted(range(len(metric_data)), key=lambda i: metric_data[i]["iou"])
    chosen = [ranked[-1], ranked[len(ranked) // 2], ranked[2]]
    labels = ["Image", "Ground truth", "Baseline U-Net", "Improved model"]
    grid = Image.new("RGB", (4 * 300, 3 * 300 + 70), "white")
    draw = ImageDraw.Draw(grid)
    for col, label in enumerate(labels):
        draw.text((col * 300 + 18, 18), label, fill="black", font=font(22, True))
    gif_frames = []
    for row, idx in enumerate(chosen):
        image_t, mask_t, _ = ds[idx]
        source = inverse_image(image_t)
        truth = Image.fromarray((mask_t[0].numpy() > 0.5).astype(np.uint8) * 255)
        with torch.inference_mode():
            preds = {k: (torch.sigmoid(m(image_t[None].to(device))["mask"])[0, 0].cpu().numpy() >= 0.5) for k, m in models.items()}
        tiles = [source, overlay(source, truth, (40, 190, 90))]
        tiles.extend(overlay(source, Image.fromarray(preds[k].astype(np.uint8) * 255), COLORS[k]) for k in ("baseline", "improved"))
        for col, tile in enumerate(tiles):
            tile = tile.resize((280, 280), Image.Resampling.BILINEAR)
            grid.paste(tile, (col * 300 + 10, row * 300 + 70))
        if row == 1:
            for text, tile in zip(labels, tiles):
                frame = Image.new("RGB", (960, 720), (248, 248, 248))
                td = ImageDraw.Draw(frame)
                td.text((30, 25), "wjw Image Processing - Pedestrian Segmentation", fill="black", font=font(30, True))
                td.text((30, 75), text, fill=(40, 40, 40), font=font(26))
                frame.paste(tile.resize((600, 600), Image.Resampling.BILINEAR), (180, 110))
                gif_frames.append(frame)
    grid.save(OUT / "qualitative_comparison.png")
    gif_frames[0].save(OUT / "demo.gif", save_all=True, append_images=gif_frames[1:], duration=1500, loop=0)


if __name__ == "__main__":
    plot_history()
    predict_models()
    print(f"Figures written to {OUT}")


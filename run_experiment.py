from __future__ import annotations

import argparse
import subprocess
import sys


def run(args):
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=128)
    args = p.parse_args()
    py = sys.executable
    for variant in ("baseline", "improved"):
        run([py, "train.py", "--variant", variant, "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--image-size", str(args.image_size)])
        run([py, "evaluate.py", "--variant", variant, "--checkpoint", f"runs/{variant}/checkpoints/best.pt", "--output", f"runs/{variant}/test_metrics.json", "--batch-size", str(args.batch_size), "--image-size", str(args.image_size)])
    run([py, "scripts/make_figures.py"])


if __name__ == "__main__":
    main()


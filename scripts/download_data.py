from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "downloads" / "PennFudanPed.zip"
TARGET = ROOT / "data" / "raw"


def main() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    TARGET.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL) as response, ARCHIVE.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"Archive SHA-256: {digest}")
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(TARGET)
    images = list((TARGET / "PennFudanPed" / "PNGImages").glob("*.png"))
    masks = list((TARGET / "PennFudanPed" / "PedMasks").glob("*.png"))
    if len(images) != 170 or len(masks) != 170:
        raise RuntimeError(f"Dataset validation failed: {len(images)} images, {len(masks)} masks")
    print(f"Ready: {len(images)} images and {len(masks)} masks in {TARGET / 'PennFudanPed'}")


if __name__ == "__main__":
    main()


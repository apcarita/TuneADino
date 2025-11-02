#!/usr/bin/env python3
"""
Scan a dataset folder for zero-byte and unreadable image files.
Optionally move them to a quarantine folder so training won't crash.

Usage:
  python sanitize_dataset.py /path/to/images [--quarantine /path/to/quarantine]

Notes:
- Zero-byte files are always considered bad.
- Unreadable files are those PIL cannot open/verify.
- Uses PIL.Image.verify() to avoid full decode.
"""
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")

def find_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(ext))
    return files


def scan_images(paths: List[Path]) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    zero: List[Path] = []
    bad: List[Tuple[Path, str]] = []
    for p in paths:
        try:
            if p.stat().st_size == 0:
                zero.append(p)
                continue
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append((p, str(e)))
    return zero, bad


def quarantine(files: List[Path], qdir: Path) -> None:
    qdir.mkdir(parents=True, exist_ok=True)
    for p in files:
        rel = p.name  # keep flat to avoid deep trees
        dest = qdir / rel
        # Ensure unique name if collision
        i = 1
        base = dest.stem
        suf = dest.suffix
        while dest.exists():
            dest = qdir / f"{base}__{i}{suf}"
            i += 1
        try:
            shutil.move(str(p), str(dest))
        except Exception as e:
            print(f"Failed to move {p} -> {dest}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Dataset root directory")
    ap.add_argument("--quarantine", type=Path, default=None, help="Directory to move bad files to")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    print(f"Scanning for images under: {root}")
    paths = find_images(root)
    print(f"Found {len(paths)} image-like files")

    zero, bad = scan_images(paths)

    # De-duplicate: zero-byte are a subset of bad in some cases, but keep both lists
    bad_only = [p for p, _ in bad if p not in set(zero)]

    print(f"Zero-byte files: {len(zero)}")
    if zero:
        for p in zero[:10]:
            print(f"  0B: {p}")
        if len(zero) > 10:
            print(f"  ... and {len(zero)-10} more")

    print(f"Unreadable files (PIL): {len(bad_only)}")
    if bad_only:
        for p in bad_only[:10]:
            print(f"  BAD: {p}")
        if len(bad_only) > 10:
            print(f"  ... and {len(bad_only)-10} more")

    if args.quarantine:
        qdir = args.quarantine
        print(f"Quarantining {len(zero)} zero-byte and {len(bad_only)} unreadable files into: {qdir}")
        quarantine(zero + bad_only, qdir)
        print("Quarantine complete.")

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def split_file(src: Path, out_dir: Path, part_size: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: List[Path] = []
    idx = 1
    with src.open("rb") as f:
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            part_path = out_dir / f"{src.name}.part{idx:02d}"
            with part_path.open("wb") as pf:
                pf.write(chunk)
            parts.append(part_path)
            idx += 1
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split large checkpoint into git-friendly parts")
    parser.add_argument("--src", required=True, help="Path to source checkpoint")
    parser.add_argument("--out-dir", required=True, help="Output directory for parts")
    parser.add_argument("--part-size-mb", type=int, default=95, help="Chunk size in MB")
    parser.add_argument("--run-id", required=True, help="W&B run id")
    parser.add_argument("--run-url", required=True, help="W&B run url")
    parser.add_argument("--best-val-f1", default="", help="Best validation F1 string")
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    part_size = args.part_size_mb * 1024 * 1024

    parts = split_file(src=src, out_dir=out_dir, part_size=part_size)

    manifest = {
        "model_filename": src.name,
        "full_size_bytes": src.stat().st_size,
        "full_md5": md5_file(src),
        "source": {
            "wandb_run_id": args.run_id,
            "wandb_run_url": args.run_url,
            "best_val_f1": args.best_val_f1,
        },
        "parts": [
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "md5": md5_file(p),
            }
            for p in parts
        ],
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(parts)} parts and manifest: {manifest_path}")


if __name__ == "__main__":
    main()

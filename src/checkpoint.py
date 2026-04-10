from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _validate_part(part_path: Path, expected_md5: str | None) -> None:
    if not part_path.exists():
        raise FileNotFoundError(f"Missing checkpoint part: {part_path}")
    if expected_md5:
        actual = file_md5(part_path)
        if actual != expected_md5:
            raise ValueError(
                f"Part checksum mismatch for {part_path.name}: {actual} != {expected_md5}"
            )


def read_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assemble_checkpoint_from_parts(
    parts_dir: Path,
    output_path: Path,
    manifest_path: Path,
) -> Path:
    manifest = read_manifest(manifest_path)
    parts: List[Dict] = manifest["parts"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out:
        for part in parts:
            part_name = part["name"]
            part_path = parts_dir / part_name
            _validate_part(part_path, part.get("md5"))
            with part_path.open("rb") as pf:
                while True:
                    buf = pf.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)

    expected_md5 = manifest.get("full_md5")
    if expected_md5:
        actual_md5 = file_md5(output_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Full checkpoint checksum mismatch: {actual_md5} != {expected_md5}"
            )

    return output_path


def ensure_checkpoint(checkpoints_dir: Path, output_dir: Path) -> Path:
    manifest_path = checkpoints_dir / "manifest.json"
    manifest = read_manifest(manifest_path)
    model_filename = manifest["model_filename"]
    output_path = output_dir / model_filename

    if output_path.exists() and manifest.get("full_md5"):
        if file_md5(output_path) == manifest["full_md5"]:
            return output_path

    return assemble_checkpoint_from_parts(
        parts_dir=checkpoints_dir,
        output_path=output_path,
        manifest_path=manifest_path,
    )

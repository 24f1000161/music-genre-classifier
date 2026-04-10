from __future__ import annotations

import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError("HF_TOKEN (or equivalent) is required")

    repo_id = os.getenv("HF_SPACE_REPO_ID", "24f1000161/music-genre-classifier-space")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    hf_url = f"https://huggingface.co/spaces/{repo_id}.git"
    # Configure remote idempotently.
    run(["git", "remote", "remove", "hf"], cwd=repo_root) if _remote_exists(repo_root, "hf") else None
    run(["git", "remote", "add", "hf", hf_url], cwd=repo_root)
    run(["git", "push", "-u", "hf", "publish-main:main"], cwd=repo_root)


def _remote_exists(repo_root: Path, name: str) -> bool:
    out = subprocess.run(
        ["git", "remote"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    remotes = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return name in remotes


if __name__ == "__main__":
    main()

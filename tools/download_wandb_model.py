from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description='Download a file from a specific W&B run')
    parser.add_argument('--run-path', required=True, help='entity/project/run_id')
    parser.add_argument('--file-name', required=True, help='file name inside the run files')
    parser.add_argument('--out-dir', default='models', help='output directory')
    args = parser.parse_args()

    api = wandb.Api(timeout=120)
    run = api.run(args.run_path)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run.file(args.file_name).download(root=args.out_dir, replace=True)
    print(f'Downloaded {args.file_name} from {run.url} to {args.out_dir}')


if __name__ == '__main__':
    main()

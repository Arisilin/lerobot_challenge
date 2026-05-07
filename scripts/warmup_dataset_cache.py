#!/usr/bin/env python
"""
离线预热 HuggingFace datasets 的 parquet 缓存。

第一次用 LeRobotDataset 加载数据时会读入所有 parquet 并写入 HF 缓存，耗时会很长。
本脚本用与训练时相同的 repo_id/root 先跑一遍加载，把缓存建好；
之后正式训练时会直接命中缓存，启动会快很多。

用法（与 train 脚本里的 --dataset.repo_id / --dataset.root 保持一致）:
  python scripts/warmup_dataset_cache.py --repo_id=./data/leju_task3 --root=./data/leju_task3

可选:
  --num_proc=4  并行读 parquet，加快首次建缓存
"""
import argparse
import sys
from pathlib import Path

# 保证能 import lerobot（添加 src 到路径）
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Warm up HF dataset cache for a LeRobot dataset.")
    parser.add_argument("--repo_id", type=str, required=True, help="Same as training --dataset.repo_id")
    parser.add_argument("--root", type=str, default=None, help="Same as training --dataset.root (default: repo_id)")
    parser.add_argument("--num_proc", type=int, default=None, help="Parallel processes for parquet load (default: 1)")
    args = parser.parse_args()
    root = args.root if args.root is not None else args.repo_id

    print("Loading dataset (this builds the cache)...")
    print(f"  repo_id={args.repo_id}, root={root}, num_proc={args.num_proc}")
    ds = LeRobotDataset(args.repo_id, root=root, num_proc=args.num_proc)
    _ = ds.hf_dataset
    print("Done. Cache is ready; training will use it on next run.")


if __name__ == "__main__":
    main()

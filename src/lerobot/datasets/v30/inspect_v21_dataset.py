#!/usr/bin/env python
"""
干读 v2.1 数据集的 parquet 和 meta 信息，不依赖转换逻辑。
用法:
  python src/lerobot/datasets/v30/inspect_v21_dataset.py --root=/path/to/dataset/root

其中 root 为数据集根目录（其下应有 data/, meta/ 等）。
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Inspect v2.1 dataset parquet and meta.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (contains data/, meta/).")
    args = parser.parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    print("=" * 60)
    print("META / INFO")
    print("=" * 60)
    for info_name in ["meta/info.json", "info.json"]:
        p = root / info_name
        if p.exists():
            info = load_json(p)
            print(f"Loaded: {p}")
            print(json.dumps(info, indent=2, ensure_ascii=False)[:4000])
            if len(json.dumps(info)) > 4000:
                print("... (truncated)")
            break
    else:
        print("No info.json or meta/info.json found.")

    print()
    print("=" * 60)
    print("LEGACY META (episodes / tasks)")
    print("=" * 60)
    for name, rel in [("meta/episodes.jsonl", "meta/episodes.jsonl"), ("meta/tasks.jsonl", "meta/tasks.jsonl")]:
        p = root / rel
        if p.exists():
            lines = p.read_text(encoding="utf-8").strip().split("\n")[:3]
            print(f"--- {name} (first {len(lines)} lines) ---")
            for line in lines:
                print(line[:500])
        else:
            print(f"--- {name} not found ---")

    print()
    print("=" * 60)
    print("DATA PARQUET (first file schema + types)")
    print("=" * 60)
    data_dir = root / "data"
    if not data_dir.is_dir():
        print("No data/ directory.")
        return
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    if not ep_paths:
        print("No parquet files under data/.")
        return
    first = ep_paths[0]
    print(f"First file: {first}")
    table = pq.read_table(first)
    schema = table.schema
    nfields = len(schema)
    print("Schema (Arrow):")
    for i in range(nfields):
        f = schema.field(i)
        print(f"  {f.name}: {f.type}")
    print(f"Num rows: {table.num_rows}")
    # 每列实际类型（第一行）
    if table.num_rows > 0:
        print("First row column types (Python):")
        for i in range(nfields):
            col = table.column(i)
            val = col[0]
            try:
                t = type(val).__name__
                if hasattr(val, "shape"):
                    t += f" shape={getattr(val, 'shape', None)}"
                print(f"  {schema.field(i).name}: {t}")
            except Exception as e:
                print(f"  {schema.field(i).name}: (error) {e}")

    # Image 列在 parquet 里具体存的是啥（第一行）
    if table.num_rows > 0:
        print()
        print("Image columns content (first row) — what is actually stored in parquet:")
        for i in range(nfields):
            name = schema.field(i).name
            if "image" in name.lower() or schema.field(i).type.equals(pa.struct({"bytes": pa.binary(), "path": pa.string()})):
                col = table.column(i)
                val = col[0]
                try:
                    if hasattr(val, "as_py"):
                        val = val.as_py()
                    if isinstance(val, dict):
                        bytes_len = len(val.get("bytes") or b"")
                        path = val.get("path") or "(none)"
                        print(f"  {name}: path={path!r}, bytes len={bytes_len}")
                    else:
                        print(f"  {name}: {type(val).__name__} = {repr(val)[:200]}")
                except Exception as e:
                    print(f"  {name}: (error) {e}")

    print()
    print("=" * 60)
    print("IMAGES/ directory (if exists)")
    print("=" * 60)
    images_dir = root / "images"
    if images_dir.is_dir():
        for sub in sorted(images_dir.iterdir())[:10]:
            if sub.is_dir():
                eps = list(sub.iterdir())
                sample = eps[0] if eps else None
                frames = list(sample.iterdir())[:3] if sample and sample.is_dir() else []
                print(f"  {sub.name}/  (episodes: {len(eps)})")
                if sample:
                    print(f"    e.g. {sample.name}/ -> {[f.name for f in frames]}")
    else:
        print("  No images/ directory.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

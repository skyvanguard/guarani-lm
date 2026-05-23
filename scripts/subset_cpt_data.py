"""Create a deterministic 30% subset of cpt_train.jsonl for the v3 CPT refinement run."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cpt_train.jsonl")
    parser.add_argument("--output", default="data/processed/cpt_train_v3_30pct.jsonl")
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = len(lines)
    print(f"  Total records: {n:,}")

    rng = random.Random(args.seed)
    rng.shuffle(lines)
    keep = int(n * args.fraction)
    subset = lines[:keep]

    print(f"  Keeping {keep:,} ({args.fraction * 100:.1f}%)")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(subset)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

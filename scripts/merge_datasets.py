"""Merge all processed datasets into final train/val/test splits for GuaraniLM.

Combines:
  - CPT data: wikipedia + culturax + parallel text + augmented NLLB
  - SFT data: instruction samples

Creates:
  - data/processed/cpt_train.jsonl, cpt_val.jsonl, cpt_test.jsonl
  - data/processed/sft_train.jsonl, sft_val.jsonl, sft_test.jsonl

Split ratio: 90/5/5 (train/val/test)
Reports token counts and dataset statistics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

PROJECT_ROOT = _SCRIPT_DIR.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# CPT source files
CPT_SOURCES = {
    "wikipedia": INTERIM_DIR / "wikipedia_gn.jsonl",
    "culturax": INTERIM_DIR / "culturax_gn.jsonl",
    "parallel": INTERIM_DIR / "parallel_gn_es.jsonl",
    "augmented": INTERIM_DIR / "augmented_nllb.jsonl",
}

# SFT source files
SFT_SOURCES = {
    "instructions": PROCESSED_DIR / "instructions.jsonl",
}

# Default split ratios
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records: list[dict] = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_data(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test sets."""
    n = len(records)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    # Ensure at least 1 sample in each split
    if n < 3:
        return records, records, records

    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]

    return train, val, test


# ---------------------------------------------------------------------------
# Token counting (approximate)
# ---------------------------------------------------------------------------


def count_tokens_approx(records: list[dict]) -> int:
    """Approximate token count by whitespace splitting.

    For CPT records, counts the 'text' field (or 'gn' + 'es' for parallel).
    For SFT records, counts all message contents.
    """
    total = 0

    for record in records:
        if "messages" in record:
            # SFT format
            for msg in record["messages"]:
                content = msg.get("content", "")
                total += len(content.split())
        elif "text" in record:
            total += len(record["text"].split())
        else:
            # Parallel pairs or other
            for key in ("gn", "es", "text"):
                val = record.get(key, "")
                if val:
                    total += len(val.split())

    return total


def try_exact_token_count(records: list[dict], max_sample: int = 5000) -> int | None:
    """Try to count tokens using the Qwen tokenizer. Returns None if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    except Exception:
        return None

    # Sample if dataset is large
    sample = records[:max_sample] if len(records) > max_sample else records
    total = 0

    for record in sample:
        texts = []
        if "messages" in record:
            for msg in record["messages"]:
                texts.append(msg.get("content", ""))
        elif "text" in record:
            texts.append(record["text"])
        else:
            for key in ("gn", "es"):
                if key in record:
                    texts.append(record[key])

        for text in texts:
            if text:
                total += len(tokenizer.encode(text, add_special_tokens=False))

    # Extrapolate if sampled
    if len(records) > max_sample:
        total = int(total * (len(records) / max_sample))

    return total


# ---------------------------------------------------------------------------
# CPT data preparation
# ---------------------------------------------------------------------------


def prepare_cpt_record(record: dict) -> dict:
    """Normalize a record into CPT format: {"text": "..."}."""
    if "text" in record:
        return {"text": record["text"]}

    # Parallel pairs -> concatenate as bilingual text
    parts = []
    if "gn" in record:
        parts.append(record["gn"])
    if "es" in record:
        parts.append(record["es"])

    if parts:
        return {"text": "\n".join(parts)}

    # Fallback: serialize the whole record
    return {"text": json.dumps(record, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combina todos los datasets procesados en splits train/val/test."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directorio de salida (default: {PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Ratio de train split (default: {TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Ratio de validation split (default: {VAL_RATIO}).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help=f"Ratio de test split (default: {TEST_RATIO}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42).",
    )
    parser.add_argument(
        "--count-tokens",
        action="store_true",
        help="Contar tokens exactos con el tokenizer Qwen (mas lento).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"[error] Los ratios deben sumar 1.0 (actual: {total_ratio:.2f})")
        sys.exit(1)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("GuaraniLM - Merge de datasets")
    print(f"  Output: {out_dir}")
    print(f"  Split: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")

    # -----------------------------------------------------------------------
    # CPT datasets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CONTINUAL PRE-TRAINING (CPT)")
    print("=" * 60)

    cpt_all: list[dict] = []
    source_counts: dict[str, int] = {}

    for name, path in CPT_SOURCES.items():
        records = load_jsonl(path)
        source_counts[name] = len(records)
        if records:
            print(f"  {name}: {len(records):,} registros ({path.name})")
            cpt_all.extend(prepare_cpt_record(r) for r in records)
        else:
            print(f"  {name}: [no encontrado] ({path})")

    if cpt_all:
        random.shuffle(cpt_all)
        cpt_train, cpt_val, cpt_test = split_data(
            cpt_all, args.train_ratio, args.val_ratio, args.test_ratio
        )

        write_jsonl(cpt_train, out_dir / "cpt_train.jsonl")
        write_jsonl(cpt_val, out_dir / "cpt_val.jsonl")
        write_jsonl(cpt_test, out_dir / "cpt_test.jsonl")

        print(f"\n  CPT splits:")
        print(f"    Train: {len(cpt_train):,}")
        print(f"    Val:   {len(cpt_val):,}")
        print(f"    Test:  {len(cpt_test):,}")
        print(f"    Total: {len(cpt_all):,}")

        # Token counts
        approx_tokens = count_tokens_approx(cpt_all)
        print(f"\n  Tokens CPT (approx words): ~{approx_tokens:,}")

        if args.count_tokens:
            exact = try_exact_token_count(cpt_all)
            if exact is not None:
                print(f"  Tokens CPT (Qwen tokenizer): ~{exact:,}")
    else:
        print("\n  [warn] No se encontraron datos CPT.")

    # -----------------------------------------------------------------------
    # SFT datasets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUPERVISED FINE-TUNING (SFT)")
    print("=" * 60)

    sft_all: list[dict] = []

    for name, path in SFT_SOURCES.items():
        records = load_jsonl(path)
        if records:
            print(f"  {name}: {len(records):,} registros ({path.name})")
            sft_all.extend(records)
        else:
            print(f"  {name}: [no encontrado] ({path})")

    if sft_all:
        random.shuffle(sft_all)
        sft_train, sft_val, sft_test = split_data(
            sft_all, args.train_ratio, args.val_ratio, args.test_ratio
        )

        write_jsonl(sft_train, out_dir / "sft_train.jsonl")
        write_jsonl(sft_val, out_dir / "sft_val.jsonl")
        write_jsonl(sft_test, out_dir / "sft_test.jsonl")

        print(f"\n  SFT splits:")
        print(f"    Train: {len(sft_train):,}")
        print(f"    Val:   {len(sft_val):,}")
        print(f"    Test:  {len(sft_test):,}")
        print(f"    Total: {len(sft_all):,}")

        # Token counts
        approx_tokens = count_tokens_approx(sft_all)
        print(f"\n  Tokens SFT (approx words): ~{approx_tokens:,}")

        if args.count_tokens:
            exact = try_exact_token_count(sft_all)
            if exact is not None:
                print(f"  Tokens SFT (Qwen tokenizer): ~{exact:,}")
    else:
        print("\n  [warn] No se encontraron datos SFT.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)

    total_records = len(cpt_all) + len(sft_all)
    print(f"\n  Registros totales: {total_records:,}")
    print(f"    CPT: {len(cpt_all):,}")
    print(f"    SFT: {len(sft_all):,}")

    print(f"\n  Fuentes CPT:")
    for name, count in source_counts.items():
        print(f"    {name}: {count:,}")

    print(f"\n  Archivos generados en {out_dir}:")
    for f in sorted(out_dir.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1 << 20)
        # Count lines
        with open(f, "r", encoding="utf-8") as fh:
            n_lines = sum(1 for _ in fh)
        print(f"    {f.name}: {n_lines:,} registros ({size_mb:.1f} MB)")

    print("\n  Merge completado.")


if __name__ == "__main__":
    main()

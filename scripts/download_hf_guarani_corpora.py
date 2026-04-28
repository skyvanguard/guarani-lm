"""
Download large-scale Guaraní text corpora from HuggingFace.

Sources:
  1. FineTranslations (HuggingFaceFW/finetranslations) — Trillion-token multilingual
     parallel translations, filter for Guaraní (grn/gug)
  2. mOSCAR (oscar-corpus/mOSCAR) — 163-language web corpus, filter for Guaraní
  3. OSCAR-2301 (oscar-corpus/OSCAR-2301) — Web text corpus with Guaraní subset
  4. HPLT v2 — High Performance Language Technologies, Guaraní subset

Uses streaming mode to avoid downloading entire datasets.
"""
import json
import os
import sys
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package required. Install with: pip install datasets")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "hf_corpora"


def download_finetranslations(max_records=None):
    """Download Guaraní subset from FineTranslations."""
    print("\n" + "=" * 60)
    print("Downloading FineTranslations - Guaraní subset")
    print("=" * 60)

    outdir = OUTPUT_DIR / "finetranslations"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "guarani.jsonl"

    if outfile.exists():
        lines = sum(1 for _ in open(outfile, encoding="utf-8"))
        print(f"  Already exists with {lines} records. Skipping.")
        return lines

    count = 0
    try:
        # FineTranslations uses {lang}_{script} format
        for lang_code in ["gug_Latn", "grn_Latn"]:
            print(f"  Trying language code: {lang_code}")
            try:
                ds = load_dataset(
                    "HuggingFaceFW/finetranslations",
                    lang_code,
                    split="train",
                    streaming=True,


                )

                with open(outfile, "a", encoding="utf-8") as f:
                    for row in ds:
                        record = dict(row)
                        record["source_dataset"] = "finetranslations"
                        record["lang_code"] = lang_code
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

                        if count % 1000 == 0:
                            print(f"    Downloaded {count} records...")

                        if max_records and count >= max_records:
                            break

                print(f"  Got {count} records from {lang_code}")
                if max_records and count >= max_records:
                    break

            except Exception as e:
                print(f"    Config '{lang_code}' not available: {e}")
                continue

    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"  Total FineTranslations records: {count}")
    return count


def download_oscar(max_records=None):
    """Download Guaraní subset from OSCAR corpus."""
    print("\n" + "=" * 60)
    print("Downloading OSCAR - Guaraní subset")
    print("=" * 60)

    outdir = OUTPUT_DIR / "oscar"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "guarani.jsonl"

    if outfile.exists():
        lines = sum(1 for _ in open(outfile, encoding="utf-8"))
        print(f"  Already exists with {lines} records. Skipping.")
        return lines

    count = 0
    # Try different OSCAR versions and language codes
    oscar_configs = [
        ("oscar-corpus/mOSCAR", "grn_Latn", "train"),
    ]

    for dataset_name, lang_code, split in oscar_configs:
        print(f"  Trying {dataset_name} [{lang_code}]...")
        try:
            ds = load_dataset(
                dataset_name,
                lang_code,
                split=split,
                streaming=True,


            )

            with open(outfile, "a", encoding="utf-8") as f:
                for row in ds:
                    record = dict(row)
                    record["source_dataset"] = dataset_name
                    record["lang_code"] = lang_code
                    # Extract text field (varies by dataset)
                    text = record.get("text", record.get("content", ""))
                    # mOSCAR returns text as list of dicts with 'text' key
                    if isinstance(text, list):
                        parts = []
                        for item in text:
                            if isinstance(item, dict):
                                parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                parts.append(item)
                        text = "\n".join(parts)
                    if text and len(text.strip()) > 20:
                        f.write(json.dumps({
                            "text": text.strip(),
                            "source": dataset_name,
                            "lang": lang_code,
                        }, ensure_ascii=False) + "\n")
                        count += 1

                    if count % 1000 == 0:
                        print(f"    Downloaded {count} records...")

                    if max_records and count >= max_records:
                        break

            print(f"  Got {count} total records so far")
            if max_records and count >= max_records:
                break

        except Exception as e:
            print(f"    Not available: {e}")
            continue

    print(f"  Total OSCAR records: {count}")
    return count


def download_hplt(max_records=None):
    """Download Guaraní subset from HPLT corpus."""
    print("\n" + "=" * 60)
    print("Downloading HPLT v2 - Guaraní subset")
    print("=" * 60)

    outdir = OUTPUT_DIR / "hplt"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "guarani.jsonl"

    if outfile.exists():
        lines = sum(1 for _ in open(outfile, encoding="utf-8"))
        print(f"  Already exists with {lines} records. Skipping.")
        return lines

    count = 0
    hplt_configs = [
        ("HPLT/HPLT2.0_cleaned", "grn_Latn"),
    ]

    for dataset_name, lang_code in hplt_configs:
        print(f"  Trying {dataset_name} [{lang_code}]...")
        try:
            ds = load_dataset(
                dataset_name,
                lang_code,
                split="train",
                streaming=True,


            )

            with open(outfile, "a", encoding="utf-8") as f:
                for row in ds:
                    text = row.get("text", "")
                    if text and len(text.strip()) > 20:
                        f.write(json.dumps({
                            "text": text.strip(),
                            "source": dataset_name,
                            "lang": lang_code,
                        }, ensure_ascii=False) + "\n")
                        count += 1

                    if count % 1000 == 0:
                        print(f"    Downloaded {count} records...")

                    if max_records and count >= max_records:
                        break

            print(f"  Got {count} total records")
            if max_records and count >= max_records:
                break

        except Exception as e:
            print(f"    Not available: {e}")
            continue

    print(f"  Total HPLT records: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download Guaraní corpora from HuggingFace"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Max records per source (default: all)")
    parser.add_argument("--sources", nargs="+",
                        choices=["finetranslations", "oscar", "hplt", "all"],
                        default=["all"],
                        help="Which sources to download")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = args.sources
    if "all" in sources:
        sources = ["finetranslations", "oscar", "hplt"]

    total = 0

    if "finetranslations" in sources:
        total += download_finetranslations(args.max_records)

    if "oscar" in sources:
        total += download_oscar(args.max_records)

    if "hplt" in sources:
        total += download_hplt(args.max_records)

    print(f"\n{'=' * 60}")
    print(f"TOTAL RECORDS DOWNLOADED: {total:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

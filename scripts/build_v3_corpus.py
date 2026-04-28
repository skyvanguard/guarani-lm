"""
Build unified v3 corpus from all filtered Guaraní sources.

Steps:
  1. Load all sources (filtered web data + curated sources)
  2. Normalize and deduplicate
  3. Output unified corpus with stats

Usage:
  python scripts/build_v3_corpus.py
"""
import json
import hashlib
import re
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def normalize_text(text):
    """Normalize text for dedup comparison."""
    t = text.lower().strip()
    t = re.sub(r'\s+', ' ', t)
    return t


def text_hash(text, length=500):
    """Hash first N chars of normalized text for fast dedup."""
    norm = normalize_text(text)[:length]
    return hashlib.md5(norm.encode('utf-8')).hexdigest()


def load_jsonl(filepath, text_field="text", source_name=None):
    """Load records from JSONL, yielding (text, metadata) tuples."""
    if not filepath.exists():
        print(f"  SKIP (not found): {filepath}")
        return

    count = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r.get(text_field, "")
            if text and len(text.strip()) >= 50:
                meta = {
                    "source": source_name or r.get("source", "unknown"),
                }
                # Preserve useful metadata
                for key in ["titulo", "title", "category", "ft_lang", "ft_conf"]:
                    if key in r:
                        meta[key] = r[key]
                yield text.strip(), meta
                count += 1

    print(f"  Loaded {count:,} from {filepath.name}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building v3 Guaraní corpus")
    print("=" * 60)

    # Collect all sources
    all_records = []
    source_stats = defaultdict(lambda: {"count": 0, "chars": 0})

    sources = [
        # Filtered web data
        (PROCESSED_DIR / "hplt_filtered.jsonl", "text", "hplt"),
        (PROCESSED_DIR / "oscar_filtered.jsonl", "text", "oscar"),
        (PROCESSED_DIR / "orembae_ocr_filtered.jsonl", "text", "orembae_ocr"),

        # Bible versions
        (RAW_DIR / "biblia_guarani" / "biblia_NNP2015.jsonl", "text", "bible_nnp2015"),
        (RAW_DIR / "biblia_guarani" / "biblia_GPC2006.jsonl", "text", "bible_gpc2006"),
        (RAW_DIR / "biblia_guarani" / "biblia_GDC2006.jsonl", "text", "bible_gdc2006"),

        # WOL JW (load from individual JSON files)
        # Handled separately below

        # FineTranslations (Guaraní text)
        (RAW_DIR / "hf_corpora" / "finetranslations" / "guarani.jsonl",
         "og_full_text", "finetranslations"),

        # IP Gov
        (RAW_DIR / "ip_gov_guarani" / "all_ip_gov_guarani.jsonl", "text", "ip_gov"),

        # News
        (RAW_DIR / "news_guarani" / "all_news_guarani.jsonl", "text", "news"),

        # Oremba'e other
        (RAW_DIR / "orembae" / "static_poems.jsonl", "text", "orembae_poems"),
        (RAW_DIR / "orembae" / "guarani_texts.jsonl", "text", "orembae_docs"),
    ]

    print("\nLoading sources:")
    for filepath, text_field, source_name in sources:
        for text, meta in load_jsonl(filepath, text_field, source_name):
            all_records.append((text, meta))
            source_stats[source_name]["count"] += 1
            source_stats[source_name]["chars"] += len(text)

    # Load WOL JW from individual JSON files
    wol_dir = RAW_DIR / "wol_jw"
    wol_count = 0
    if wol_dir.exists():
        for jf in sorted(wol_dir.rglob("doc_*.json")):
            with open(jf, encoding="utf-8") as f:
                r = json.load(f)
                text = r.get("text", "")
                if text and len(text.strip()) >= 50:
                    meta = {
                        "source": "wol_jw",
                        "category": r.get("category", ""),
                        "title": r.get("title", ""),
                    }
                    all_records.append((text.strip(), meta))
                    source_stats["wol_jw"]["count"] += 1
                    source_stats["wol_jw"]["chars"] += len(text.strip())
                    wol_count += 1
        print(f"  Loaded {wol_count:,} from WOL JW")

    print(f"\nTotal records before dedup: {len(all_records):,}")

    # Deduplication
    print("\nDeduplicating...")
    seen_hashes = {}
    deduped = []
    dup_count = 0
    dup_by_source = defaultdict(int)

    for text, meta in all_records:
        h = text_hash(text)
        if h in seen_hashes:
            dup_count += 1
            dup_by_source[meta["source"]] += 1
        else:
            seen_hashes[h] = meta["source"]
            deduped.append((text, meta))

    print(f"  Duplicates found: {dup_count:,}")
    if dup_by_source:
        print(f"  Duplicates by source:")
        for src, cnt in sorted(dup_by_source.items(), key=lambda x: -x[1]):
            print(f"    {src}: {cnt}")

    # Write output
    output_file = PROCESSED_DIR / "v3_corpus.jsonl"
    total_chars = 0
    final_stats = defaultdict(lambda: {"count": 0, "chars": 0})

    with open(output_file, "w", encoding="utf-8") as fout:
        for text, meta in deduped:
            record = {"text": text, "source": meta["source"], "chars": len(text)}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chars += len(text)
            final_stats[meta["source"]]["count"] += 1
            final_stats[meta["source"]]["chars"] += len(text)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"V3 CORPUS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Source':<25} {'Docs':>10} {'Chars':>15} {'~Tokens':>12}")
    print("-" * 65)
    for src in sorted(final_stats.keys()):
        s = final_stats[src]
        print(f"{src:<25} {s['count']:>10,} {s['chars']:>15,} {s['chars']//4:>12,}")
    print("-" * 65)
    print(f"{'TOTAL':<25} {len(deduped):>10,} {total_chars:>15,} {total_chars//4:>12,}")
    print(f"\nOutput: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
Filter web-crawled corpora (HPLT, mOSCAR) for actual Guaraní content.

Uses two complementary methods:
  1. fasttext language identification (lid.176.bin)
  2. Guaraní-specific character/pattern heuristics

A text passes if EITHER:
  - fasttext detects 'gn' as top language with confidence >= 0.3
  - Guaraní character density is above threshold (nasal vowels, g̃, ʼ)

Usage:
  python scripts/filter_guarani.py                    # Filter all
  python scripts/filter_guarani.py --source hplt      # Filter only HPLT
  python scripts/filter_guarani.py --source oscar      # Filter only mOSCAR
  python scripts/filter_guarani.py --min-chars 100     # Min text length
"""
import json
import re
import sys
import argparse
from pathlib import Path

import fasttext

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "hf_corpora"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Load fasttext model
MODEL_PATH = PROJECT_DIR / "lid.176.bin"
if not MODEL_PATH.exists():
    print(f"ERROR: fasttext model not found at {MODEL_PATH}")
    print("Download from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
    sys.exit(1)

# Suppress fasttext warning
fasttext.FastText.eprint = lambda x: None
MODEL = fasttext.load_model(str(MODEL_PATH))

# Guaraní-specific characters
GN_NASAL_VOWELS = set("ãẽĩõũỹÃẼĨÕŨỸ")
GN_SPECIAL = set("ñÑ")
GN_GLOTTAL = set("ʼ")  # modifier letter apostrophe (puso)
# g̃ is g + combining tilde (U+0303)
GN_COMBINING_TILDE = "\u0303"

# Common Guaraní function words (very distinctive)
GN_WORDS = {
    "ha", "pe", "ko", "la", "kue", "gui", "rehe", "ndive",
    "hag̃ua", "rupi", "oĩ", "ndaipóri", "avei", "katu",
    "ñande", "ore", "pee", "hikuái", "chupe", "ichupe",
    "oiko", "ojapo", "omombe'u", "oñe'ẽ", "ohecha",
    "porã", "vai", "guasu", "michĩ", "pyahu",
    "opáichagua", "opaite", "heta", "mbovymi",
    "ága", "ko'ãga", "upéi", "upépe", "ápe",
}


def guarani_char_density(text):
    """Calculate density of Guaraní-specific characters in text."""
    if not text:
        return 0.0

    total = len(text)
    gn_chars = 0

    for ch in text:
        if ch in GN_NASAL_VOWELS:
            gn_chars += 3  # Weight nasal vowels heavily
        elif ch in GN_GLOTTAL:
            gn_chars += 3
        elif ch in GN_SPECIAL:
            gn_chars += 1

    # Check for combining tilde (g̃)
    gn_chars += text.count(GN_COMBINING_TILDE) * 3

    return gn_chars / total if total > 0 else 0.0


def guarani_word_score(text):
    """Score text based on presence of common Guaraní words."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    matches = words & GN_WORDS
    return len(matches) / max(len(GN_WORDS), 1)


def is_guarani(text, ft_threshold=0.3, char_threshold=0.005, word_threshold=0.05):
    """Determine if text is primarily in Guaraní."""
    if not text or len(text.strip()) < 20:
        return False, {}

    # Clean text for fasttext (single line, no newlines)
    clean = text.replace("\n", " ").strip()[:5000]

    # Method 1: fasttext language ID
    labels, probs = MODEL.predict(clean, k=3)
    ft_lang = labels[0].replace("__label__", "")
    ft_conf = float(probs[0])

    # Method 2: Character density
    char_density = guarani_char_density(text)

    # Method 3: Word-based score
    word_score = guarani_word_score(text)

    info = {
        "ft_lang": ft_lang,
        "ft_conf": round(ft_conf, 3),
        "char_density": round(char_density, 5),
        "word_score": round(word_score, 3),
    }

    # Decision logic:
    # Pass if fasttext says Guaraní with reasonable confidence
    if ft_lang == "gn" and ft_conf >= ft_threshold:
        return True, info

    # Pass if strong Guaraní character signal even if fasttext disagrees
    if char_density >= char_threshold and word_score >= word_threshold:
        return True, info

    # Pass if fasttext says Guaraní with low confidence but chars confirm
    if ft_lang == "gn" and char_density >= char_threshold * 0.5:
        return True, info

    return False, info


def filter_corpus(input_file, output_file, source_name, min_chars=50):
    """Filter a JSONL corpus for Guaraní content."""
    print(f"\nFiltering {source_name}: {input_file}")

    total = 0
    kept = 0
    total_chars_in = 0
    total_chars_out = 0
    rejected_langs = {}

    with open(input_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            record = json.loads(line)
            text = record.get("text", "")

            if not text or len(text.strip()) < min_chars:
                continue

            total_chars_in += len(text)

            is_gn, info = is_guarani(text)

            if is_gn:
                # Write filtered record
                out_record = {
                    "text": text.strip(),
                    "source": source_name,
                    "ft_lang": info["ft_lang"],
                    "ft_conf": info["ft_conf"],
                    "char_density": info["char_density"],
                    "chars": len(text.strip()),
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                kept += 1
                total_chars_out += len(text.strip())
            else:
                lang = info.get("ft_lang", "unknown")
                rejected_langs[lang] = rejected_langs.get(lang, 0) + 1

            if total % 10000 == 0:
                pct = kept / total * 100 if total > 0 else 0
                print(f"  {total:,} processed, {kept:,} kept ({pct:.1f}%)", flush=True)

    # Summary
    pct = kept / total * 100 if total > 0 else 0
    print(f"\n  {source_name} RESULTS:")
    print(f"    Total records: {total:,}")
    print(f"    Kept (Guaraní): {kept:,} ({pct:.1f}%)")
    print(f"    Chars in: {total_chars_in:,}")
    print(f"    Chars out: {total_chars_out:,}")
    print(f"    Tokens out: ~{total_chars_out // 4:,}")

    # Top rejected languages
    top_rejected = sorted(rejected_langs.items(), key=lambda x: -x[1])[:10]
    if top_rejected:
        print(f"    Top rejected languages:")
        for lang, count in top_rejected:
            print(f"      {lang}: {count:,}")

    return kept, total_chars_out


def main():
    parser = argparse.ArgumentParser(description="Filter corpora for Guaraní")
    parser.add_argument("--source", choices=["hplt", "oscar", "all"], default="all")
    parser.add_argument("--min-chars", type=int, default=50,
                        help="Minimum text length to consider")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = []
    if args.source in ("hplt", "all"):
        f = DATA_DIR / "hplt" / "guarani.jsonl"
        if f.exists():
            sources.append(("hplt", f, OUTPUT_DIR / "hplt_filtered.jsonl"))
    if args.source in ("oscar", "all"):
        f = DATA_DIR / "oscar" / "guarani.jsonl"
        if f.exists():
            sources.append(("oscar", f, OUTPUT_DIR / "oscar_filtered.jsonl"))

    if not sources:
        print("No source files found")
        return

    grand_kept = 0
    grand_chars = 0

    for name, input_file, output_file in sources:
        kept, chars = filter_corpus(input_file, output_file, name, args.min_chars)
        grand_kept += kept
        grand_chars += chars

    print(f"\n{'=' * 60}")
    print(f"TOTAL FILTERED:")
    print(f"  Records kept: {grand_kept:,}")
    print(f"  Characters: {grand_chars:,}")
    print(f"  Tokens: ~{grand_chars // 4:,}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Analyze contamination patterns in cpt_train and sft_v2_train datasets.

Counts:
- URLs (http, https, t.co, www, .com, .org, etc)
- Hashtags (#word)
- Twitter handles (@word)
- Emojis
- Non-Guarani/Spanish script characters (Cyrillic, CJK, Thai, Vietnamese tones)
- Excessive whitespace
- HTML tags

Reports per-file statistics + a few example contaminated records.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# --- Patterns ---
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|edu|gov|io|co|ly|tv)\b/?\S*", re.IGNORECASE)
TCO_PATTERN = re.compile(r"\bt\.co/\S+", re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r"(?:^|\s)#[A-Za-zÀ-ÿ0-9_]+")
HANDLE_PATTERN = re.compile(r"(?:^|\s)@[A-Za-z0-9_]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "♀-♂"
    "☀-⭕"
    "]",
    flags=re.UNICODE,
)
HTML_PATTERN = re.compile(r"<[^>]+>")
CJK_PATTERN = re.compile(r"[一-鿿぀-ゟ゠-ヿ]")  # CJK + Hiragana + Katakana
THAI_PATTERN = re.compile(r"[฀-๿]")
ARABIC_PATTERN = re.compile(r"[؀-ۿ]")
CYRILLIC_PATTERN = re.compile(r"[Ѐ-ӿ]")
VIETNAMESE_TONE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ]|[̀-ͯ]")  # combining marks


def analyze_text(text: str) -> dict[str, int]:
    """Return contamination counts for a single text."""
    return {
        "urls": len(URL_PATTERN.findall(text)),
        "tco": len(TCO_PATTERN.findall(text)),
        "hashtags": len(HASHTAG_PATTERN.findall(text)),
        "handles": len(HANDLE_PATTERN.findall(text)),
        "emojis": len(EMOJI_PATTERN.findall(text)),
        "html": len(HTML_PATTERN.findall(text)),
        "cjk_chars": len(CJK_PATTERN.findall(text)),
        "thai_chars": len(THAI_PATTERN.findall(text)),
        "arabic_chars": len(ARABIC_PATTERN.findall(text)),
        "cyrillic_chars": len(CYRILLIC_PATTERN.findall(text)),
        "length": len(text),
    }


def extract_text(record: dict) -> str:
    """Pull the relevant text from a CPT or SFT record."""
    if "text" in record:
        return record["text"]
    if "conversations" in record:
        return "\n".join(m.get("content", "") for m in record["conversations"])
    # Fallback: concat all string values
    return " ".join(v for v in record.values() if isinstance(v, str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample", type=int, default=0, help="Only analyze first N records (0=all)")
    parser.add_argument("--show-examples", type=int, default=3, help="Show N worst contaminated records")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Not found: {path}")

    totals = Counter()
    records_with: dict[str, int] = Counter()
    total_records = 0
    total_chars = 0
    worst: list[tuple[int, int, str]] = []  # (score, line_no, snippet)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if args.sample and total_records >= args.sample:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = extract_text(rec)
            stats = analyze_text(text)

            total_records += 1
            total_chars += stats["length"]
            for k, v in stats.items():
                if k == "length":
                    continue
                totals[k] += v
                if v > 0:
                    records_with[k] += 1

            # Track worst-contaminated (excluding length)
            score = (
                stats["urls"] * 3 + stats["tco"] * 5 + stats["hashtags"] * 2 +
                stats["handles"] * 2 + stats["emojis"] +
                stats["cjk_chars"] + stats["thai_chars"] +
                stats["arabic_chars"] + stats["cyrillic_chars"]
            )
            if score > 0:
                snippet = text[:200].replace("\n", " ")
                worst.append((score, line_no, snippet))

    worst.sort(reverse=True)

    print(f"\n=== Contamination report: {path} ===")
    print(f"Records analyzed: {total_records:,}")
    print(f"Total characters: {total_chars:,}")
    print()
    print(f"{'Pattern':<18} {'Total occurrences':>18} {'Records with':>15} {'% records':>11}")
    print("-" * 65)
    for key in ["urls", "tco", "hashtags", "handles", "emojis", "html",
                "cjk_chars", "thai_chars", "arabic_chars", "cyrillic_chars"]:
        pct = (records_with[key] / total_records * 100) if total_records else 0
        print(f"{key:<18} {totals[key]:>18,} {records_with[key]:>15,} {pct:>10.2f}%")

    if args.show_examples and worst:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(f"\n=== Top {args.show_examples} contaminated records ===")
        for score, line_no, snippet in worst[:args.show_examples]:
            print(f"\n[line {line_no}, score={score}]")
            print(f"  {snippet}")


if __name__ == "__main__":
    main()

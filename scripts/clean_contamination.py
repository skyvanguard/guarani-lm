"""Clean contamination from cpt_train and sft_v2_train datasets.

Two-stage cleaning:
1. Inline replacement: remove URLs, handles, hashtags, emojis, HTML.
2. Record filtering: drop records that become too short, or have too much
   residual foreign script (CJK, Thai, Arabic, Cyrillic, Vietnamese tones).

Outputs *_clean.jsonl and prints stats on what was dropped / modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --- Cleaning patterns (broader than analyze, to catch variants) ---
URL_PATTERN = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|\b[a-z0-9-]+\.(?:com|net|org|edu|gov|io|co|ly|tv|info|me|us|uk|es|py|ar|br)\b/?\S*",
    re.IGNORECASE,
)
TCO_PATTERN = re.compile(r"\bt\.co/\S+", re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ0-9])#[A-Za-zÀ-ÿ0-9_]+")
HANDLE_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÿ0-9])@[A-Za-z0-9_]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]",
    flags=re.UNICODE,
)
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINES_PATTERN = re.compile(r"\n{3,}")

# Foreign-script detection (used for filtering, not removal)
FOREIGN_SCRIPT = re.compile(r"[一-鿿぀-ゟ゠-ヿ฀-๿؀-ۿЀ-ӿ]")


def clean_text(text: str) -> str:
    """Apply inline cleaning to a text string."""
    text = TCO_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = HANDLE_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub("", text)
    text = HTML_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = NEWLINES_PATTERN.sub("\n\n", text)
    return text.strip()


def should_drop(
    cleaned: str,
    *,
    min_length: int,
    max_foreign_ratio: float,
) -> tuple[bool, str]:
    """Return (drop, reason)."""
    if len(cleaned) < min_length:
        return True, f"too_short(<{min_length})"

    foreign_count = len(FOREIGN_SCRIPT.findall(cleaned))
    if foreign_count > 0:
        # Cheap heuristic: foreign characters per total characters
        ratio = foreign_count / max(len(cleaned), 1)
        if ratio > max_foreign_ratio:
            return True, f"foreign_script({ratio:.3f}>{max_foreign_ratio})"

    return False, "kept"


def process_cpt_record(rec: dict) -> dict | None:
    text = rec.get("text", "")
    cleaned = clean_text(text)
    rec["text"] = cleaned
    return rec


def process_sft_record(rec: dict) -> dict | None:
    """Clean the content of each conversation turn."""
    conversations = rec.get("conversations")
    if not conversations:
        return rec
    new_convs = []
    for turn in conversations:
        content = turn.get("content", "")
        cleaned = clean_text(content)
        new_turn = dict(turn)
        new_turn["content"] = cleaned
        new_convs.append(new_turn)
    rec["conversations"] = new_convs
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["cpt", "sft"], required=True,
                        help="cpt = plain {text:...}, sft = {conversations:[{role,content}]}")
    parser.add_argument("--min-length", type=int, default=100,
                        help="Drop records shorter than this after cleaning (CPT) or where any turn is shorter (SFT)")
    parser.add_argument("--max-foreign-ratio", type=float, default=0.02,
                        help="Drop if foreign-script chars > this fraction of text")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        sys.exit(f"Not found: {in_path}")

    n_in = 0
    n_kept = 0
    n_modified = 0
    drop_reasons: dict[str, int] = {}

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                drop_reasons["bad_json"] = drop_reasons.get("bad_json", 0) + 1
                continue

            if args.mode == "cpt":
                original_text = rec.get("text", "")
                cleaned_rec = process_cpt_record(dict(rec))
                if cleaned_rec is None:
                    drop_reasons["process_failed"] = drop_reasons.get("process_failed", 0) + 1
                    continue
                final_text = cleaned_rec["text"]
                drop, reason = should_drop(
                    final_text,
                    min_length=args.min_length,
                    max_foreign_ratio=args.max_foreign_ratio,
                )
                if drop:
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue
                if final_text != original_text:
                    n_modified += 1
                fout.write(json.dumps(cleaned_rec, ensure_ascii=False) + "\n")
                n_kept += 1

            else:  # sft
                cleaned_rec = process_sft_record(dict(rec))
                if cleaned_rec is None or "conversations" not in cleaned_rec:
                    drop_reasons["no_conversations"] = drop_reasons.get("no_conversations", 0) + 1
                    continue
                # Concatenate all turns for the length/foreign check
                concat = "\n".join(t.get("content", "") for t in cleaned_rec["conversations"])
                drop, reason = should_drop(
                    concat,
                    min_length=args.min_length,
                    max_foreign_ratio=args.max_foreign_ratio,
                )
                if drop:
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue
                fout.write(json.dumps(cleaned_rec, ensure_ascii=False) + "\n")
                n_kept += 1

    print(f"\n=== Clean report: {in_path} -> {out_path} ===")
    print(f"Records in:       {n_in:,}")
    print(f"Records kept:     {n_kept:,} ({n_kept / n_in * 100:.2f}%)")
    if args.mode == "cpt":
        print(f"Records modified: {n_modified:,}  (of those kept)")
    print(f"Records dropped:  {n_in - n_kept:,} ({(n_in - n_kept) / n_in * 100:.2f}%)")
    if drop_reasons:
        print(f"\nDrop reasons:")
        for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30} {count:>10,}")


if __name__ == "__main__":
    main()

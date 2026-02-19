"""Count actual words/tokens across all downloaded datasets."""
import json
import csv
import os
import bz2
import lzma
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def count_words(text: str) -> int:
    return len(text.split())

def count_jsonl(path: Path, text_fields: list[str]) -> dict:
    words = 0
    records = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            for field in text_fields:
                if field in obj:
                    val = obj[field]
                    if isinstance(val, str):
                        words += count_words(val)
    return {"records": records, "words": words}

def count_csv(path: Path, text_cols: list[str] | None = None) -> dict:
    words = 0
    records = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records += 1
            if text_cols:
                for col in text_cols:
                    if col in row and row[col]:
                        words += count_words(row[col])
            else:
                for val in row.values():
                    if val:
                        words += count_words(val)
    return {"records": records, "words": words}

def count_compressed_text(path: Path) -> dict:
    words = 0
    lines = 0
    opener = bz2.open if path.suffix == ".bz2" else lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                lines += 1
                words += count_words(line)
    return {"records": lines, "words": words}

def count_tsv_bz2(path: Path) -> dict:
    words = 0
    records = 0
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            records += 1
            for p in parts:
                words += count_words(p)
    return {"records": records, "words": words}

def count_xml_bz2(path: Path) -> dict:
    """Rough word count from Wikipedia XML dump."""
    words = 0
    pages = 0
    import re
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        in_text = False
        for line in f:
            if "<text" in line:
                in_text = True
                # Get text after the tag
                idx = line.find(">")
                if idx >= 0:
                    words += count_words(line[idx+1:])
            elif "</text>" in line:
                in_text = False
                idx = line.find("</text>")
                words += count_words(line[:idx])
            elif in_text:
                words += count_words(line)
            if "<title>" in line:
                pages += 1
    return {"records": pages, "words": words}

def count_gongora_tweets(gongora_dir: Path) -> dict:
    """Count words in Gongora Tweets_set directory (CSVs with Tweet column)."""
    words = 0
    records = 0
    tweets_dir = gongora_dir / "Tweets_set"
    if tweets_dir.exists():
        for csv_file in sorted(tweets_dir.glob("*.csv")):
            with open(csv_file, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    tweet = row.get("Tweet", "")
                    if tweet:
                        records += 1
                        words += count_words(tweet)
    return {"records": records, "words": words}


def main():
    results = {}
    total_words = 0
    total_records = 0

    # === JSONL files ===
    jsonl_configs = {
        "HPLT 2.0": ("hplt2/hplt2_cleaned.jsonl", ["text"]),
        "Alpaca Guarani": ("alpaca_guarani/alpaca_guarani.jsonl", ["instruction", "input", "output"]),
        "Jojajovai HF": ("jojajovai_hf/jojajovai_hf.jsonl", ["gn", "es"]),
        "Wikipedia HF": ("wikipedia_hf/wikipedia_gn.jsonl", ["text"]),
        "Sentiment": ("mmaguero/sentiment.jsonl", ["text"]),
        "Offensive": ("mmaguero/offensive.jsonl", ["text"]),
        "Humor": ("mmaguero/humor.jsonl", ["text"]),
        "Emotion": ("mmaguero/emotion.jsonl", ["text"]),
        "GuaSpa 2023": ("gua_spa/gua_spa_2023.jsonl", ["text"]),
        "FLORES+": ("eval/flores200.jsonl", ["text"]),
        "Belebele": ("eval/belebele.jsonl", ["flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"]),
    }

    for name, (rel_path, fields) in jsonl_configs.items():
        path = RAW_DIR / rel_path
        if path.exists():
            r = count_jsonl(path, fields)
            results[name] = r
            total_words += r["words"]
            total_records += r["records"]
        else:
            results[name] = {"records": 0, "words": 0, "note": "NOT FOUND"}

    # === CSV files ===
    csv_configs = {
        "Jojajovai CSV": ("jojajovai/jojajovai_all.csv", None),
        "Gov Traductor ES-GN": ("traductor_gov_es_gn.csv", None),
        "Gov Vocabulario GN-ES": ("traductor_gov_gn_es.csv", None),
    }

    for name, (rel_path, cols) in csv_configs.items():
        path = RAW_DIR / rel_path
        if path.exists():
            r = count_csv(path, cols)
            results[name] = r
            total_words += r["words"]
            total_records += r["records"]
        else:
            results[name] = {"records": 0, "words": 0, "note": "NOT FOUND"}

    # === Compressed files ===
    cc100_path = RAW_DIR / "cc100_gn.txt.xz"
    if cc100_path.exists():
        r = count_compressed_text(cc100_path)
        results["CC-100 GN"] = r
        total_words += r["words"]
        total_records += r["records"]

    tatoeba_path = RAW_DIR / "tatoeba_grn.tsv.bz2"
    if tatoeba_path.exists():
        r = count_tsv_bz2(tatoeba_path)
        results["Tatoeba GN"] = r
        total_words += r["words"]
        total_records += r["records"]

    wiki_xml = RAW_DIR / "gnwiki-latest-pages-articles.xml.bz2"
    if wiki_xml.exists():
        r = count_xml_bz2(wiki_xml)
        results["Wikipedia XML dump"] = r
        total_words += r["words"]
        total_records += r["records"]

    # === Gongora tweets ===
    gongora_main = RAW_DIR / "gongora" / "giossa-gongora-guarani-2021-main"
    if gongora_main.exists():
        r = count_gongora_tweets(gongora_main)
        results["Gongora Tweets"] = r
        total_words += r["words"]
        total_records += r["records"]

    # === Leipzig corpus (tar.gz -> extract and count) ===
    leipzig_tar = RAW_DIR / "grn_community_2017.tar.gz"
    if leipzig_tar.exists():
        import tarfile
        words = 0
        records = 0
        with tarfile.open(leipzig_tar, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("-sentences.txt"):
                    f = tar.extractfile(member)
                    if f:
                        for line in f:
                            line = line.decode("utf-8", errors="replace").strip()
                            if line:
                                records += 1
                                # Leipzig format: ID\tsentence
                                parts = line.split("\t", 1)
                                if len(parts) > 1:
                                    words += count_words(parts[1])
        results["Leipzig GN"] = {"records": records, "words": words}
        total_words += words
        total_records += records

    # === Gongora parallel CSVs ===
    gongora_parallel = RAW_DIR / "gongora" / "giossa-gongora-guarani-2021-main"
    if gongora_parallel.exists():
        words = 0
        records = 0
        for csv_file in gongora_parallel.glob("*.csv"):
            try:
                r = count_csv(csv_file)
                words += r["words"]
                records += r["records"]
            except Exception:
                pass
        if words > 0:
            results["Gongora CSVs"] = {"records": records, "words": words}
            total_words += words
            total_records += records

    # === Print results ===
    print("\n" + "=" * 70)
    print(f"{'DATASET':<30} {'REGISTROS':>12} {'PALABRAS':>12} {'~TOKENS':>12}")
    print("=" * 70)

    for name, r in sorted(results.items(), key=lambda x: x[1].get("words", 0), reverse=True):
        rec = r.get("records", 0)
        w = r.get("words", 0)
        tok = int(w * 1.3)
        note = r.get("note", "")
        suffix = f"  ({note})" if note else ""
        print(f"{name:<30} {rec:>12,} {w:>12,} {tok:>12,}{suffix}")

    print("-" * 70)
    est_tokens = int(total_words * 1.3)
    print(f"{'TOTAL':<30} {total_records:>12,} {total_words:>12,} {est_tokens:>12,}")
    print("=" * 70)
    print(f"\nNota: ~tokens = palabras x 1.3 (factor promedio subword para Guarani)")
    print(f"Esto es una estimacion. El tokenizer real (Qwen2.5) puede variar.")


if __name__ == "__main__":
    main()

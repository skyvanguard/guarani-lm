"""Merge all processed data into final train/val/test splits.

Produces:
  - data/processed/cpt_train.jsonl  — Continual Pre-Training data
  - data/processed/cpt_val.jsonl    — CPT validation
  - data/processed/sft_train.jsonl  — SFT instruction data
  - data/processed/sft_val.jsonl    — SFT validation
  - data/processed/test.jsonl       — Held-out test set (CPT + eval benchmarks)

CPT data sources (with upsampling weights):
  - Wikipedia (cleaned)         weight=3  (high quality)
  - HPLT 2.0                   weight=1  (web crawl)
  - CC-100                      weight=1  (web crawl)
  - Leipzig                     weight=2  (curated sentences)
  - Parallel (gn+es sides)      weight=2  (bilingual)
  - Gongora Tweets              weight=1  (social media)
  - Alpaca (text portions)      weight=1  (instruction text)
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import tarfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42


def _text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def load_jsonl(path: Path, text_field: str = "text", min_words: int = 5) -> list[dict]:
    records = []
    if not path.exists():
        print(f"  [warn] {path.name} no encontrado, saltando.")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field, "")
            if text and len(text.split()) >= min_words:
                records.append(obj)
    return records


def load_compressed_text(path: Path) -> list[dict]:
    import bz2
    import lzma

    records = []
    if not path.exists():
        print(f"  [warn] {path.name} no encontrado, saltando.")
        return records

    opener = bz2.open if path.suffix == ".bz2" else lzma.open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if text and len(text.split()) >= 5:
                records.append({"text": text, "source": path.stem})
    return records


def upsample(records: list[dict], weight: int) -> list[dict]:
    if weight <= 1:
        return list(records)
    return records * weight


def deduplicate(records: list[dict], text_field: str = "text") -> list[dict]:
    seen = set()
    unique = []
    for rec in records:
        h = _text_hash(rec.get(text_field, ""))
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    return unique


def split_data(
    records: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = records[:n_test]
    val = records[n_test : n_test + n_val]
    train = records[n_test + n_val :]
    return train, val, test


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    print("=" * 60)
    print("  GuaraniLM - Merge Datasets")
    print("=" * 60)

    # ===== CPT DATA =====
    print("\n--- CPT: Cargando fuentes ---")
    cpt_records: list[dict] = []

    # 1. Wikipedia cleaned
    wiki_path = PROCESSED_DIR / "wikipedia_clean.jsonl"
    wiki = load_jsonl(wiki_path, "text")
    print(f"  Wikipedia:     {len(wiki):>8,} registros (weight=3)")
    cpt_records.extend(upsample(wiki, 3))

    # 2. HPLT 2.0
    hplt_path = RAW_DIR / "hplt2" / "hplt2_cleaned.jsonl"
    hplt = load_jsonl(hplt_path, "text")
    print(f"  HPLT 2.0:      {len(hplt):>8,} registros (weight=1)")
    cpt_records.extend(hplt)

    # 3. CC-100
    cc100_path = RAW_DIR / "cc100_gn.txt.xz"
    cc100 = load_compressed_text(cc100_path)
    print(f"  CC-100:        {len(cc100):>8,} registros (weight=1)")
    cpt_records.extend(cc100)

    # 4. Leipzig
    leipzig: list[dict] = []
    leipzig_path = RAW_DIR / "grn_community_2017.tar.gz"
    if leipzig_path.exists():
        with tarfile.open(leipzig_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("-sentences.txt"):
                    f = tar.extractfile(member)
                    if f:
                        for line in f:
                            text = line.decode("utf-8", errors="replace").strip()
                            parts = text.split("\t", 1)
                            if len(parts) > 1 and len(parts[1].split()) >= 5:
                                leipzig.append({"text": parts[1], "source": "leipzig"})
    print(f"  Leipzig:       {len(leipzig):>8,} registros (weight=2)")
    cpt_records.extend(upsample(leipzig, 2))

    # 5. Parallel data (both sides for CPT)
    parallel_path = PROCESSED_DIR / "parallel_all.jsonl"
    parallel: list[dict] = []
    if parallel_path.exists():
        with open(parallel_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gn = obj.get("gn", "")
                if gn and len(gn.split()) >= 3:
                    parallel.append({"text": gn, "source": "parallel_gn"})
                es = obj.get("es", "")
                if es and len(es.split()) >= 3:
                    parallel.append({"text": es, "source": "parallel_es"})
    else:
        print(f"  [warn] parallel_all.jsonl no existe. Ejecuta prepare_parallel.py primero.")
    print(f"  Parallel:      {len(parallel):>8,} registros (weight=2)")
    cpt_records.extend(upsample(parallel, 2))

    # 6. Gongora Tweets
    tweets: list[dict] = []
    tweets_dir = RAW_DIR / "gongora" / "giossa-gongora-guarani-2021-main" / "Tweets_set"
    if tweets_dir.exists():
        for csv_file in sorted(tweets_dir.glob("*.csv")):
            with open(csv_file, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    text = row.get("Tweet", "")
                    if text and len(text.split()) >= 3:
                        tweets.append({"text": text, "source": "gongora_tweets"})
    print(f"  Tweets:        {len(tweets):>8,} registros (weight=1)")
    cpt_records.extend(tweets)

    # 7. Alpaca (monolingual portions)
    alpaca_path = RAW_DIR / "alpaca_guarani" / "alpaca_guarani.jsonl"
    alpaca: list[dict] = []
    if alpaca_path.exists():
        with open(alpaca_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parts = []
                for field in ["instruction", "input", "output"]:
                    v = obj.get(field, "")
                    if v:
                        parts.append(v)
                text = " ".join(parts)
                if len(text.split()) >= 5:
                    alpaca.append({"text": text, "source": "alpaca_guarani"})
    print(f"  Alpaca:        {len(alpaca):>8,} registros (weight=1)")
    cpt_records.extend(alpaca)

    # Deduplicate CPT
    print(f"\n  CPT antes de dedup: {len(cpt_records):,}")
    cpt_records = deduplicate(cpt_records)
    print(f"  CPT despues de dedup: {len(cpt_records):,}")

    # Split CPT
    cpt_train, cpt_val, cpt_test = split_data(cpt_records, VAL_RATIO, TEST_RATIO, SEED)

    write_jsonl(cpt_train, PROCESSED_DIR / "cpt_train.jsonl")
    write_jsonl(cpt_val, PROCESSED_DIR / "cpt_val.jsonl")
    print(f"  CPT train: {len(cpt_train):,}  |  val: {len(cpt_val):,}")

    # ===== SFT DATA =====
    print("\n--- SFT: Cargando instrucciones ---")
    sft_path = PROCESSED_DIR / "sft_all.jsonl"
    sft_train = []
    sft_val = []
    if sft_path.exists():
        sft_records = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sft_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        print(f"  SFT total: {len(sft_records):,}")

        rng = random.Random(SEED)
        rng.shuffle(sft_records)
        n_sft_val = int(len(sft_records) * VAL_RATIO)
        sft_val = sft_records[:n_sft_val]
        sft_train = sft_records[n_sft_val:]

        write_jsonl(sft_train, PROCESSED_DIR / "sft_train.jsonl")
        write_jsonl(sft_val, PROCESSED_DIR / "sft_val.jsonl")
        print(f"  SFT train: {len(sft_train):,}  |  val: {len(sft_val):,}")
    else:
        print(f"  [warn] {sft_path.name} no existe. Ejecuta prepare_instructions.py primero.")

    # ===== TEST SET =====
    test_records = cpt_test
    for eval_file in ["flores200.jsonl", "belebele.jsonl"]:
        eval_path = RAW_DIR / "eval" / eval_file
        if eval_path.exists():
            eval_data = load_jsonl(eval_path, "text", min_words=0)
            test_records.extend(eval_data)
            print(f"  Eval {eval_file}: {len(eval_data):,} registros agregados a test")

    write_jsonl(test_records, PROCESSED_DIR / "test.jsonl")
    print(f"  Test total: {len(test_records):,}")

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    total_cpt_words = sum(len(r.get("text", "").split()) for r in cpt_train)
    est_cpt_tokens = int(total_cpt_words * 1.3)
    print(f"  CPT train:  {len(cpt_train):>10,} registros  ~{est_cpt_tokens:>12,} tokens")
    if sft_train:
        print(f"  SFT train:  {len(sft_train):>10,} registros")
    print(f"  Val (CPT):  {len(cpt_val):>10,} registros")
    if sft_val:
        print(f"  Val (SFT):  {len(sft_val):>10,} registros")
    print(f"  Test:       {len(test_records):>10,} registros")
    print(f"\n  Archivos en: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Unify all parallel (bilingual gn<->es) data into one JSONL file.

Reads from multiple sources in data/raw/, normalizes Guarani text,
deduplicates, filters, and writes to data/processed/parallel_all.jsonl.

Sources:
  - Jojajovai CSV  (jojajovai/jojajovai_all.csv)
  - Jojajovai HF   (jojajovai_hf/jojajovai_hf.jsonl)
  - Gov ES->GN      (traductor_gov_es_gn.csv)
  - Gov GN->ES      (traductor_gov_gn_es.csv)
  - Tatoeba          (tatoeba_grn.tsv.bz2) — monolingual gn only
  - GuaSpa 2023      (gua_spa/gua_spa_2023.jsonl) — monolingual gn (NER data)
  - Gongora           (gongora/...ParallelSet/*.aligned inside zips)

NOT included:
  - FLORES+ (eval/flores200.jsonl) — reserved for evaluation
  - Alpaca Guarani — instruction data, not simple parallel
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize  # noqa: E402

PROJECT_ROOT = _SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "parallel_all.jsonl"

# Minimum word count for either side of a parallel pair
MIN_WORDS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash(text: str) -> str:
    """Return a short hash of the text for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Source readers — each returns list[dict] with keys gn, es (optional), source
# ---------------------------------------------------------------------------


def read_jojajovai_csv() -> list[dict]:
    """Jojajovai CSV — columns: split,source,gn,es,tokens_gn,tokens_es."""
    path = RAW_DIR / "jojajovai" / "jojajovai_all.csv"
    if not path.exists():
        print(f"  [skip] No encontrado: {path}")
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gn = (row.get("gn") or "").strip()
            es = (row.get("es") or "").strip()
            if gn and es:
                records.append({"gn": gn, "es": es, "source": "jojajovai"})

    print(f"  jojajovai_csv: {len(records)} pares leidos")
    return records


def read_jojajovai_hf() -> list[dict]:
    """Jojajovai HF JSONL — fields: gn, es."""
    path = RAW_DIR / "jojajovai_hf" / "jojajovai_hf.jsonl"
    if not path.exists():
        print(f"  [skip] No encontrado: {path}")
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            gn = (obj.get("gn") or "").strip()
            es = (obj.get("es") or "").strip()
            if gn and es:
                records.append({"gn": gn, "es": es, "source": "jojajovai"})

    print(f"  jojajovai_hf: {len(records)} pares leidos")
    return records


def read_gov_es_gn() -> list[dict]:
    """Gov traductor ES->GN — columns: palabra_id, palabra(ES), clavebusqueda, significado(GN), archivo."""
    path = RAW_DIR / "traductor_gov_es_gn.csv"
    if not path.exists():
        print(f"  [skip] No encontrado: {path}")
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            es = (row.get("palabra") or "").strip()
            gn = (row.get("significado") or "").strip()
            if gn and es and gn.upper() != "NULL":
                records.append({"gn": gn, "es": es, "source": "gov"})

    print(f"  gov_es_gn: {len(records)} pares leidos")
    return records


def read_gov_gn_es() -> list[dict]:
    """Gov vocabulario GN->ES — columns: palabra_id, palabra(GN), clavebusqueda, significado(ES), archivo."""
    path = RAW_DIR / "traductor_gov_gn_es.csv"
    if not path.exists():
        print(f"  [skip] No encontrado: {path}")
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gn = (row.get("palabra") or "").strip()
            es = (row.get("significado") or "").strip()
            if gn and es and es.upper() != "NULL":
                records.append({"gn": gn, "es": es, "source": "gov"})

    print(f"  gov_gn_es: {len(records)} pares leidos")
    return records


def read_tatoeba() -> list[dict]:
    """Tatoeba GRN — bz2 compressed TSV: ID\\tlang\\ttext. Monolingual gn only."""
    path = RAW_DIR / "tatoeba_grn.tsv.bz2"
    if not path.exists():
        print(f"  [skip] No encontrado: {path}")
        return []

    records: list[dict] = []
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[1] == "grn":
                gn = parts[2].strip()
                if gn:
                    records.append({"gn": gn, "source": "tatoeba"})

    print(f"  tatoeba: {len(records)} oraciones monolingues leidas")
    return records


def read_guaspa() -> list[dict]:
    """GuaSpa 2023 NER JSONL — field 'text' contains Guarani. Monolingual."""
    path = RAW_DIR / "gua_spa" / "gua_spa_2023.jsonl"
    if not path.exists():
        print(f"  [skip] No encontrado: {path}")
        return []

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            gn = (obj.get("text") or "").strip()
            if gn:
                records.append({"gn": gn, "source": "guaspa"})

    print(f"  guaspa: {len(records)} oraciones monolingues leidas")
    return records


def _read_aligned_content(content: str) -> list[tuple[str, str]]:
    """Parse a .aligned file content into (gn, es) pairs.

    Format: blocks separated by blank lines.  Each block has lines
    starting with 'gn: ' and 'es: '.
    """
    pairs: list[tuple[str, str]] = []
    blocks = content.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        gn_text = ""
        es_text = ""
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("gn:"):
                gn_text = line[3:].strip()
            elif line.startswith("es:"):
                es_text = line[3:].strip()
        if gn_text and es_text:
            pairs.append((gn_text, es_text))
    return pairs


def read_gongora() -> list[dict]:
    """Gongora ParallelSet — .aligned files inside two zip archives."""
    parallel_dir = RAW_DIR / "gongora" / "giossa-gongora-guarani-2021-main" / "ParallelSet"
    if not parallel_dir.exists():
        print(f"  [skip] No encontrado: {parallel_dir}")
        return []

    records: list[dict] = []
    zip_files = sorted(parallel_dir.glob("*.zip"))

    if not zip_files:
        print(f"  [skip] No hay archivos .zip en {parallel_dir}")
        return []

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                aligned_names = [n for n in zf.namelist() if n.endswith(".aligned")]
                for name in aligned_names:
                    with zf.open(name) as f:
                        content = f.read().decode("utf-8", errors="replace")
                    pairs = _read_aligned_content(content)
                    for gn, es in pairs:
                        records.append({"gn": gn, "es": es, "source": "gongora"})
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  [warn] Error leyendo {zip_path.name}: {exc}")

    print(f"  gongora: {len(records)} pares leidos")
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("GuaraniLM - Preparacion corpus paralelo unificado")
    print("=" * 60)
    print()

    # ── 1. Read all sources ──────────────────────────────────────
    print("Leyendo fuentes...")
    all_records: list[dict] = []
    stats: dict[str, dict[str, int]] = {}

    # Parallel (bilingual) sources
    for reader_fn, label in [
        (read_jojajovai_csv, "jojajovai_csv"),
        (read_jojajovai_hf, "jojajovai_hf"),
        (read_gov_es_gn, "gov_es_gn"),
        (read_gov_gn_es, "gov_gn_es"),
        (read_gongora, "gongora"),
    ]:
        recs = reader_fn()
        stats[label] = {"raw": len(recs)}
        all_records.extend(recs)

    # Monolingual sources
    for reader_fn, label in [
        (read_tatoeba, "tatoeba"),
        (read_guaspa, "guaspa"),
    ]:
        recs = reader_fn()
        stats[label] = {"raw": len(recs)}
        all_records.extend(recs)

    total_raw = len(all_records)
    print(f"\n  Total registros crudos: {total_raw}")

    # ── 2. Normalize, filter, deduplicate ────────────────────────
    print("\nNormalizando y filtrando...")
    seen_hashes: set[str] = set()
    kept: list[dict] = []
    dupes = 0
    filtered_short = 0
    source_kept: dict[str, int] = {}

    for rec in all_records:
        # Normalize Guarani side
        gn = normalize(rec["gn"])
        es = rec.get("es", "")

        # Spanish side: basic strip (normalize only applies to gn)
        if es:
            es = es.strip()

        # Filter: skip if gn is empty or too short
        if not gn or _word_count(gn) < MIN_WORDS:
            filtered_short += 1
            continue

        # For bilingual records, also check es side
        is_bilingual = bool(es)
        if is_bilingual and _word_count(es) < MIN_WORDS:
            filtered_short += 1
            continue

        # Deduplicate by hash of normalized gn text
        h = _hash(gn.lower())
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)

        # Build output record
        out: dict[str, str] = {"gn": gn, "source": rec["source"]}
        if is_bilingual:
            out["es"] = es

        kept.append(out)
        source_kept[rec["source"]] = source_kept.get(rec["source"], 0) + 1

    # ── 3. Write output ──────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for rec in kept:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 4. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Resumen")
    print("=" * 60)

    print(f"\n{'Fuente':<20} {'Leidos':>10} {'Escritos':>10}")
    print("-" * 42)
    for label in stats:
        raw = stats[label]["raw"]
        # Map label to source name used in records
        source_name_map = {
            "jojajovai_csv": "jojajovai",
            "jojajovai_hf": "jojajovai",
            "gov_es_gn": "gov",
            "gov_gn_es": "gov",
            "gongora": "gongora",
            "tatoeba": "tatoeba",
            "guaspa": "guaspa",
        }
        print(f"  {label:<18} {raw:>10}")

    print("-" * 42)
    print(f"\n  Registros por source en salida:")
    for src, cnt in sorted(source_kept.items(), key=lambda x: -x[1]):
        print(f"    {src:<18} {cnt:>10}")

    total_kept = len(kept)
    bilingual = sum(1 for r in kept if "es" in r)
    monolingual = total_kept - bilingual

    print(f"\n  Total leidos:        {total_raw:>10}")
    print(f"  Filtrados (<{MIN_WORDS} words): {filtered_short:>10}")
    print(f"  Duplicados:          {dupes:>10}")
    print(f"  Total escritos:      {total_kept:>10}")
    print(f"    - bilingues:       {bilingual:>10}")
    print(f"    - monolingues gn:  {monolingual:>10}")

    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1 << 20)
        print(f"\n  Archivo: {OUTPUT_PATH}")
        print(f"  Tamano:  {size_mb:.2f} MB")

    print()


if __name__ == "__main__":
    main()

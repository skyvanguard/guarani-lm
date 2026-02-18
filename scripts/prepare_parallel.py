"""Prepare the Jojajovai parallel corpus for training.

Reads CSV/TSV files from the Jojajovai dataset (data/raw/jojajovai/),
creates aligned Guarani-Spanish parallel pairs, and applies normalization.

Output: data/interim/parallel_gn_es.jsonl  (one {"gn": "...", "es": "..."} per pair)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "jojajovai"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "parallel_gn_es.jsonl"

# Minimum length for either side of a parallel pair
MIN_PAIR_LENGTH = 5

# Maximum length ratio between the two sides (avoids misaligned pairs)
MAX_LENGTH_RATIO = 5.0


# ---------------------------------------------------------------------------
# File detection and reading
# ---------------------------------------------------------------------------


def detect_delimiter(file_path: Path) -> str:
    """Detect CSV delimiter by reading the first few lines."""
    with open(file_path, "r", encoding="utf-8") as fh:
        sample = fh.read(4096)

    # Try common delimiters
    for delim in ["\t", ",", ";", "|"]:
        if delim in sample:
            # Verify it appears consistently
            lines = sample.strip().split("\n")[:5]
            counts = [line.count(delim) for line in lines if line.strip()]
            if counts and all(c == counts[0] for c in counts) and counts[0] > 0:
                return delim

    return ","  # fallback


def detect_columns(
    headers: list[str],
) -> tuple[int | None, int | None]:
    """Detect which columns contain Guarani and Spanish text.

    Returns (gn_col_index, es_col_index) or (None, None) if not found.
    """
    gn_col = None
    es_col = None

    gn_keywords = {"gn", "guarani", "guaraní", "avañe'ẽ", "source_gn", "text_gn", "guarani_text"}
    es_keywords = {"es", "spanish", "español", "castellano", "source_es", "text_es", "spanish_text"}

    headers_lower = [h.strip().lower() for h in headers]

    for i, h in enumerate(headers_lower):
        if h in gn_keywords or any(kw in h for kw in gn_keywords):
            gn_col = i
        elif h in es_keywords or any(kw in h for kw in es_keywords):
            es_col = i

    # If only two columns and no match, assume first=gn, second=es
    if gn_col is None and es_col is None and len(headers) == 2:
        gn_col, es_col = 0, 1

    return gn_col, es_col


def read_parallel_file(file_path: Path) -> list[tuple[str, str]]:
    """Read a parallel corpus file and return (guarani, spanish) pairs."""
    pairs: list[tuple[str, str]] = []

    suffix = file_path.suffix.lower()
    delimiter = "\t" if suffix == ".tsv" else detect_delimiter(file_path)

    print(f"  Leyendo {file_path.name} (delimiter={repr(delimiter)}) ...")

    with open(file_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=delimiter)

        # Try to detect header row
        first_row = next(reader, None)
        if first_row is None:
            return pairs

        gn_col, es_col = detect_columns(first_row)

        if gn_col is not None and es_col is not None:
            # First row was a header, columns detected
            print(f"    Columnas detectadas: gn={first_row[gn_col]}, es={first_row[es_col]}")
        else:
            # First row might be data; assume 2-column format (gn, es)
            if len(first_row) >= 2:
                gn_col, es_col = 0, 1
                # Re-add first row as data
                pairs.append((first_row[gn_col].strip(), first_row[es_col].strip()))
                print(f"    Sin header detectado, asumiendo col0=gn, col1=es")
            else:
                print(f"    [warn] No se puede determinar estructura de {file_path.name}")
                return pairs

        for row in reader:
            if len(row) <= max(gn_col, es_col):
                continue
            gn_text = row[gn_col].strip()
            es_text = row[es_col].strip()
            if gn_text and es_text:
                pairs.append((gn_text, es_text))

    return pairs


def read_jsonl_file(file_path: Path) -> list[tuple[str, str]]:
    """Read parallel pairs from a JSONL file."""
    pairs: list[tuple[str, str]] = []

    print(f"  Leyendo {file_path.name} (JSONL) ...")

    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try common field names
            gn_text = (
                record.get("gn")
                or record.get("guarani")
                or record.get("source_gn")
                or record.get("text_gn")
                or ""
            )
            es_text = (
                record.get("es")
                or record.get("spanish")
                or record.get("español")
                or record.get("source_es")
                or record.get("text_es")
                or ""
            )

            if gn_text and es_text:
                pairs.append((gn_text.strip(), es_text.strip()))

    return pairs


# ---------------------------------------------------------------------------
# Quality filters for parallel pairs
# ---------------------------------------------------------------------------


def is_valid_pair(gn: str, es: str) -> bool:
    """Check if a parallel pair passes quality filters."""
    # Minimum length
    if len(gn) < MIN_PAIR_LENGTH or len(es) < MIN_PAIR_LENGTH:
        return False

    # Length ratio check (avoids badly aligned pairs)
    ratio = max(len(gn), len(es)) / max(min(len(gn), len(es)), 1)
    if ratio > MAX_LENGTH_RATIO:
        return False

    # Skip if both sides are identical (probably not a real translation)
    if gn.lower() == es.lower():
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara el corpus paralelo Jojajovai (Guarani-Espanol)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directorio con archivos Jojajovai (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Archivo JSONL de salida (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=MIN_PAIR_LENGTH,
        help=f"Longitud minima por lado del par (default: {MIN_PAIR_LENGTH}).",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=MAX_LENGTH_RATIO,
        help=f"Ratio maximo de longitud entre lados (default: {MAX_LENGTH_RATIO}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MIN_PAIR_LENGTH, MAX_LENGTH_RATIO
    MIN_PAIR_LENGTH = args.min_length
    MAX_LENGTH_RATIO = args.max_ratio

    input_dir = args.input_dir

    if not input_dir.exists():
        print(f"[error] Directorio de entrada no encontrado: {input_dir}")
        print("  Ejecuta primero: python scripts/download_data.py --sources jojajovai")
        sys.exit(1)

    print("GuaraniLM - Preparacion corpus paralelo Jojajovai")
    print(f"  Directorio entrada: {input_dir}")

    # Collect all data files
    all_pairs: list[tuple[str, str]] = []

    supported_extensions = {".csv", ".tsv", ".txt"}
    data_files = sorted(
        f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions
    )
    jsonl_files = sorted(
        f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in {".jsonl", ".json"}
    )

    if not data_files and not jsonl_files:
        print(f"  [warn] No se encontraron archivos de datos en {input_dir}")
        # List what's there
        all_files = list(input_dir.rglob("*"))
        if all_files:
            print(f"  Archivos encontrados:")
            for f in all_files[:20]:
                print(f"    {f.relative_to(input_dir)}")
        sys.exit(1)

    for f in data_files:
        pairs = read_parallel_file(f)
        print(f"    -> {len(pairs)} pares")
        all_pairs.extend(pairs)

    for f in jsonl_files:
        pairs = read_jsonl_file(f)
        print(f"    -> {len(pairs)} pares")
        all_pairs.extend(pairs)

    print(f"\n  Total pares crudos: {len(all_pairs)}")

    # Normalize and filter
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    filtered = 0
    seen: set[str] = set()

    with open(args.output, "w", encoding="utf-8") as fout:
        for gn_raw, es_raw in tqdm(all_pairs, desc="Normalizando"):
            # Normalize Guarani side
            gn = normalize(gn_raw)
            # Spanish side: basic cleanup (strip whitespace, NFC)
            es = es_raw.strip()

            if not is_valid_pair(gn, es):
                filtered += 1
                continue

            # Dedup by Guarani text
            dedup_key = gn.lower()
            if dedup_key in seen:
                filtered += 1
                continue
            seen.add(dedup_key)

            record = {"gn": gn, "es": es}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nResultados:")
    print(f"  Pares validos: {kept}")
    print(f"  Filtrados/duplicados: {filtered}")

    if args.output.exists():
        size_mb = args.output.stat().st_size / (1 << 20)
        print(f"\nGuardado: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

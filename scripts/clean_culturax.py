"""Filter and clean the CulturaX Guarani subset for LM training.

Loads the CulturaX dataset (either from HuggingFace or a local JSONL file),
applies quality filters and deduplication, then outputs clean text.

Output: data/interim/culturax_gn.jsonl  (one {"text": "..."} per document)
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "culturax_gn" / "culturax_gn.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "culturax_gn.jsonl"

# Quality filter thresholds
MIN_TEXT_LENGTH = 50       # Minimum characters
MAX_TEXT_LENGTH = 100_000  # Maximum characters (filter boilerplate/dumps)
MIN_WORD_COUNT = 10        # Minimum words
MAX_REPETITION_RATIO = 0.3  # Max fraction of repeated lines within a document


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


def _is_too_short(text: str) -> bool:
    return len(text) < MIN_TEXT_LENGTH


def _is_too_long(text: str) -> bool:
    return len(text) > MAX_TEXT_LENGTH


def _has_too_few_words(text: str) -> bool:
    return len(text.split()) < MIN_WORD_COUNT


def _has_excessive_repetition(text: str) -> bool:
    """Check if a document has too many repeated lines."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 3:
        return False
    unique = set(lines)
    repetition_ratio = 1.0 - (len(unique) / len(lines))
    return repetition_ratio > MAX_REPETITION_RATIO


def _content_hash(text: str) -> str:
    """Compute a hash for near-exact deduplication."""
    # Normalize whitespace before hashing
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def passes_quality_filters(text: str) -> bool:
    """Return True if the text passes all quality filters."""
    if _is_too_short(text):
        return False
    if _is_too_long(text):
        return False
    if _has_too_few_words(text):
        return False
    if _has_excessive_repetition(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_from_jsonl(
    input_path: Path,
    output_path: Path,
    text_field: str = "text",
) -> dict[str, int]:
    """Process CulturaX from a local JSONL file."""
    stats = {"total": 0, "kept": 0, "filtered": 0, "duplicates": 0}
    seen_hashes: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, desc="Filtrando CulturaX"):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["filtered"] += 1
                continue

            text = record.get(text_field, "")
            if not text:
                stats["filtered"] += 1
                continue

            # Apply Guarani normalization
            text = normalize(text)

            # Quality filters
            if not passes_quality_filters(text):
                stats["filtered"] += 1
                continue

            # Deduplication
            h = _content_hash(text)
            if h in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(h)

            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


def process_from_huggingface(
    output_path: Path,
) -> dict[str, int]:
    """Load CulturaX directly from HuggingFace and process."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[error] 'datasets' no instalado. Ejecuta: pip install datasets")
        sys.exit(1)

    stats = {"total": 0, "kept": 0, "filtered": 0, "duplicates": 0}
    seen_hashes: set[str] = set()

    print("Cargando CulturaX (grn) desde HuggingFace ...")
    ds = load_dataset("uonlp/CulturaX", "grn", split="train", trust_remote_code=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(ds, desc="Filtrando CulturaX"):
            stats["total"] += 1
            text = row.get("text", "")
            if not text:
                stats["filtered"] += 1
                continue

            text = normalize(text)

            if not passes_quality_filters(text):
                stats["filtered"] += 1
                continue

            h = _content_hash(text)
            if h in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(h)

            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra y limpia el subset CulturaX Guarani."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Archivo JSONL local de CulturaX (default: data/raw/culturax_gn/culturax_gn.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Archivo JSONL de salida (default: data/interim/culturax_gn.jsonl).",
    )
    parser.add_argument(
        "--from-huggingface",
        action="store_true",
        help="Descargar directamente desde HuggingFace en vez de archivo local.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=MIN_TEXT_LENGTH,
        help=f"Longitud minima en caracteres (default: {MIN_TEXT_LENGTH}).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_TEXT_LENGTH,
        help=f"Longitud maxima en caracteres (default: {MAX_TEXT_LENGTH}).",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Campo JSON que contiene el texto (default: text).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
    MIN_TEXT_LENGTH = args.min_length
    MAX_TEXT_LENGTH = args.max_length

    print("GuaraniLM - Limpieza CulturaX Guarani")
    print(f"  Filtros: min_length={MIN_TEXT_LENGTH}, max_length={MAX_TEXT_LENGTH}")

    if args.from_huggingface:
        stats = process_from_huggingface(args.output)
    else:
        if not args.input.exists():
            print(f"[error] Archivo de entrada no encontrado: {args.input}")
            print("  Opciones:")
            print("  1. Ejecuta primero: python scripts/download_data.py --sources culturax")
            print("  2. Usa --from-huggingface para descargar directo")
            sys.exit(1)
        stats = process_from_jsonl(args.input, args.output, text_field=args.text_field)

    print(f"\nResultados:")
    print(f"  Total procesados: {stats['total']}")
    print(f"  Documentos validos: {stats['kept']}")
    print(f"  Filtrados (calidad): {stats['filtered']}")
    print(f"  Duplicados removidos: {stats['duplicates']}")

    if args.output.exists():
        size_mb = args.output.stat().st_size / (1 << 20)
        print(f"\nGuardado: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

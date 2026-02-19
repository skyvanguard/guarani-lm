"""Guarani text normalization utilities.

Provides a `normalize(text)` function that applies the full normalization
pipeline for Guarani text:
  - Unicode NFC normalization
  - Nasal vowel normalization (combining diacritics -> precomposed)
  - g-tilde (g̃) normalization
  - Puso / glottal stop normalization (various apostrophes -> U+0027)
  - Removal of non-Guarani characters
  - Whitespace collapsing

Can also process entire JSONL files from the command line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Character tables
# ---------------------------------------------------------------------------

# Combining tilde U+0303
COMBINING_TILDE = "\u0303"

# Map of base + combining tilde -> precomposed nasal vowels
_NASAL_VOWEL_MAP: dict[str, str] = {
    f"a{COMBINING_TILDE}": "\u00e3",  # a + combining tilde -> ã
    f"A{COMBINING_TILDE}": "\u00c3",  # A + combining tilde -> Ã
    f"e{COMBINING_TILDE}": "\u1ebd",  # e + combining tilde -> ẽ
    f"E{COMBINING_TILDE}": "\u1ebc",  # E + combining tilde -> Ẽ
    f"i{COMBINING_TILDE}": "\u0129",  # i + combining tilde -> ĩ
    f"I{COMBINING_TILDE}": "\u0128",  # I + combining tilde -> Ĩ
    f"o{COMBINING_TILDE}": "\u00f5",  # o + combining tilde -> õ
    f"O{COMBINING_TILDE}": "\u00d5",  # O + combining tilde -> Õ
    f"u{COMBINING_TILDE}": "\u0169",  # u + combining tilde -> ũ
    f"U{COMBINING_TILDE}": "\u0168",  # U + combining tilde -> Ũ
    f"y{COMBINING_TILDE}": "\u1ef9",  # y + combining tilde -> ỹ
    f"Y{COMBINING_TILDE}": "\u1ef8",  # Y + combining tilde -> Ỹ
}

# Apostrophe-like characters that should map to standard puso (')
_APOSTROPHE_VARIANTS: dict[str, str] = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201a": "'",  # single low-9 quotation mark
    "\u02bc": "'",  # modifier letter apostrophe
    "\u02bb": "'",  # modifier letter turned comma
    "\u02c8": "'",  # modifier letter vertical line
    "\u0060": "'",  # grave accent
    "\u00b4": "'",  # acute accent
    "\u2032": "'",  # prime
    "\u02b9": "'",  # modifier letter prime
    "\uff07": "'",  # fullwidth apostrophe
}

# Characters allowed in Guarani text:
# - Basic Latin letters (a-z, A-Z)
# - Guarani nasal vowels: ã ẽ ĩ õ ũ ỹ (and uppercase)
# - g̃ / G̃ (g + combining tilde, kept as digraph)
# - Puso: ' (U+0027)
# - Accented vowels from Spanish loanwords: á é í ó ú ñ
# - Digits, basic punctuation, whitespace
_ALLOWED_PATTERN = re.compile(
    r"[^"
    r"a-zA-Z0-9"
    r"\u00e3\u00c3"  # ã Ã
    r"\u1ebd\u1ebc"  # ẽ Ẽ
    r"\u0129\u0128"  # ĩ Ĩ
    r"\u00f5\u00d5"  # õ Õ
    r"\u0169\u0168"  # ũ Ũ
    r"\u1ef9\u1ef8"  # ỹ Ỹ
    r"\u0303"        # combining tilde (for g̃)
    r"'"             # puso / glottal stop
    r"\u00e1\u00c1"  # á Á
    r"\u00e9\u00c9"  # é É
    r"\u00ed\u00cd"  # í Í
    r"\u00f3\u00d3"  # ó Ó
    r"\u00fa\u00da"  # ú Ú
    r"\u00f1\u00d1"  # ñ Ñ
    r"\u00fc\u00dc"  # ü Ü (rare, but present in some Guarani texts)
    r"\s"            # whitespace
    r".,;:!?\-\"()\[\]{}/\\_@#$%&*+=<>|~"  # punctuation
    r"]"
)

# Collapse runs of whitespace
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Normalization pipeline
# ---------------------------------------------------------------------------


def _normalize_unicode_nfc(text: str) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize("NFC", text)


def _normalize_nasal_vowels(text: str) -> str:
    """Convert base vowel + combining tilde to precomposed nasal vowels."""
    # First decompose to NFD so all combining chars are separated
    text = unicodedata.normalize("NFD", text)

    # Replace known base+combining pairs with precomposed forms
    for decomposed, precomposed in _NASAL_VOWEL_MAP.items():
        text = text.replace(decomposed, precomposed)

    # Re-compose remaining sequences
    text = unicodedata.normalize("NFC", text)
    return text


def _normalize_g_tilde(text: str) -> str:
    """Normalize g-tilde to consistent representation (g + combining tilde).

    In Guarani, g̃ has no precomposed Unicode form, so we consistently
    represent it as g + U+0303 (combining tilde).
    """
    # Some texts may use other representations; ensure consistency.
    # After NFC, g̃ should already be g + combining tilde.
    # Handle any leftover cases.
    return text


def _normalize_puso(text: str) -> str:
    """Normalize all apostrophe variants to standard ASCII apostrophe (U+0027)."""
    for variant, replacement in _APOSTROPHE_VARIANTS.items():
        text = text.replace(variant, replacement)
    return text


def _remove_non_guarani_chars(text: str) -> str:
    """Remove characters not used in Guarani text."""
    return _ALLOWED_PATTERN.sub("", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs and excessive newlines."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def normalize(text: str) -> str:
    """Apply the full Guarani text normalization pipeline.

    Args:
        text: Raw input text.

    Returns:
        Normalized text suitable for LM training.
    """
    if not text:
        return ""

    text = _normalize_unicode_nfc(text)
    text = _normalize_nasal_vowels(text)
    text = _normalize_g_tilde(text)
    text = _normalize_puso(text)
    text = _remove_non_guarani_chars(text)
    text = _collapse_whitespace(text)

    return text


# ---------------------------------------------------------------------------
# JSONL file processing
# ---------------------------------------------------------------------------


def process_jsonl(
    input_path: Path,
    output_path: Path,
    text_field: str = "text",
    text_fields: list[str] | None = None,
    min_words: int = 0,
) -> int:
    """Normalize all text in a JSONL file.

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to write the normalized JSONL file.
        text_field: Name of the JSON field containing text to normalize.
        text_fields: List of fields to normalize (overrides text_field).
        min_words: Minimum word count to keep a record (0 = keep all).

    Returns:
        Number of records written.
    """
    fields = text_fields or [text_field]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] Linea {line_num}: JSON invalido, saltando.")
                continue

            for f in fields:
                if f in record and isinstance(record[f], str):
                    record[f] = normalize(record[f])

            # Filter short records
            if min_words > 0:
                total_words = sum(
                    len(record.get(f, "").split()) for f in fields if isinstance(record.get(f), str)
                )
                if total_words < min_words:
                    skipped += 1
                    continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if skipped:
        print(f"  [info] {skipped} registros filtrados (<{min_words} palabras)")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza texto Guarani (string individual o archivos JSONL)."
    )
    subparsers = parser.add_subparsers(dest="command", help="Modo de operacion.")

    # Sub-command: text
    text_parser = subparsers.add_parser("text", help="Normalizar un string de texto.")
    text_parser.add_argument("input_text", help="Texto a normalizar.")

    # Sub-command: file
    file_parser = subparsers.add_parser("file", help="Normalizar un archivo JSONL.")
    file_parser.add_argument("input", type=Path, help="Archivo JSONL de entrada.")
    file_parser.add_argument("output", type=Path, help="Archivo JSONL de salida.")
    file_parser.add_argument(
        "--field",
        default="text",
        help="Campo JSON que contiene el texto (default: text).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "text":
        print(normalize(args.input_text))

    elif args.command == "file":
        print(f"Procesando {args.input} -> {args.output}")
        count = process_jsonl(args.input, args.output, text_field=args.field)
        print(f"Procesados {count} registros.")

    else:
        print("Uso: normalize_guarani.py {text|file} ...")
        print("  normalize_guarani.py text \"Che sy oguata ka'aguype\"")
        print("  normalize_guarani.py file input.jsonl output.jsonl")
        sys.exit(1)


if __name__ == "__main__":
    main()

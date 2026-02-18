"""Parse and clean Guarani Wikipedia XML dump for LM training.

Reads a MediaWiki XML dump (bz2 compressed), extracts article text,
removes wiki markup, and applies Guarani normalization.

Output: data/interim/wikipedia_gn.jsonl  (one {"text": "..."} per article)
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "gnwiki-latest-pages-articles.xml.bz2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "wikipedia_gn.jsonl"

# Minimum characters after cleaning to keep an article (filters stubs)
MIN_ARTICLE_LENGTH = 100

# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

# Regex patterns for wikitext markup removal
_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
_RE_REF_SELF = re.compile(r"<ref[^/]*/\s*>")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_RE_CATEGORY = re.compile(r"\[\[(?:Category|Categor[ií]a|Ñemohenda):[^\]]*\]\]", re.IGNORECASE)
_RE_INTERWIKI = re.compile(r"\[\[[a-z]{2,3}(?:-[a-z]+)?:[^\]]*\]\]")
_RE_FILE = re.compile(
    r"\[\[(?:File|Image|Archivo|Imagen|Ta'anga):[^\]]*\]\]", re.IGNORECASE
)
_RE_LINK_PIPED = re.compile(r"\[\[[^\]]*?\|([^\]]+)\]\]")
_RE_LINK_SIMPLE = re.compile(r"\[\[([^\]]+)\]\]")
_RE_EXTERNAL_LINK_TEXT = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_RE_EXTERNAL_LINK_BARE = re.compile(r"\[https?://\S+\]")
_RE_BOLD_ITALIC = re.compile(r"'{2,5}")
_RE_HEADING = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
_RE_MAGIC_WORD = re.compile(r"__[A-Z_]+__")
_RE_LIST_BULLET = re.compile(r"^[*#;:]+\s*", re.MULTILINE)


def strip_wikitext(wikitext: str) -> str:
    """Remove MediaWiki markup from article wikitext, returning plain text."""
    text = wikitext

    # Remove HTML comments
    text = _RE_COMMENT.sub("", text)

    # Remove references
    text = _RE_REF.sub("", text)
    text = _RE_REF_SELF.sub("", text)

    # Remove tables
    text = _RE_TABLE.sub("", text)

    # Remove templates (nested — iterate until stable)
    prev = None
    while prev != text:
        prev = text
        text = _RE_TEMPLATE.sub("", text)

    # Remove files/images
    text = _RE_FILE.sub("", text)

    # Remove categories
    text = _RE_CATEGORY.sub("", text)

    # Remove interwiki links
    text = _RE_INTERWIKI.sub("", text)

    # Resolve piped links [[target|display]] -> display
    text = _RE_LINK_PIPED.sub(r"\1", text)

    # Resolve simple links [[target]] -> target
    text = _RE_LINK_SIMPLE.sub(r"\1", text)

    # External links with text [url text] -> text
    text = _RE_EXTERNAL_LINK_TEXT.sub(r"\1", text)

    # Bare external links [url] -> ""
    text = _RE_EXTERNAL_LINK_BARE.sub("", text)

    # Remove remaining HTML tags
    text = _RE_TAG.sub("", text)

    # Bold/italic
    text = _RE_BOLD_ITALIC.sub("", text)

    # Headings =...= -> text
    text = _RE_HEADING.sub(r"\1", text)

    # Magic words
    text = _RE_MAGIC_WORD.sub("", text)

    # List bullets
    text = _RE_LIST_BULLET.sub("", text)

    return text


# ---------------------------------------------------------------------------
# XML dump parsing
# ---------------------------------------------------------------------------


def parse_dump(input_path: Path) -> list[dict[str, str]]:
    """Parse a MediaWiki XML dump and extract cleaned article texts.

    Uses mwxml for robust XML parsing and wikitextparser as a fallback
    for any remaining markup.
    """
    articles: list[dict[str, str]] = []

    try:
        import mwxml
    except ImportError:
        print("[error] 'mwxml' no instalado. Ejecuta: pip install mwxml")
        sys.exit(1)

    # mwxml can handle bz2 files directly
    print(f"Parseando dump: {input_path}")
    dump = mwxml.Dump.from_file(
        bz2.open(input_path, "rt", encoding="utf-8") if input_path.suffix == ".bz2"
        else open(input_path, "r", encoding="utf-8")
    )

    # Optional: use wikitextparser for deeper cleanup
    try:
        import wikitextparser as wtp
        has_wtp = True
    except ImportError:
        has_wtp = False
        print("  [info] wikitextparser no disponible, usando regex puro.")

    skipped_redirect = 0
    skipped_short = 0
    processed = 0

    for page in dump:
        # Skip non-article namespaces (0 = main namespace)
        if page.namespace != 0:
            continue

        # Get the latest revision
        revision = None
        for rev in page:
            revision = rev

        if revision is None or revision.text is None:
            continue

        raw_text = revision.text

        # Skip redirects
        if raw_text.strip().upper().startswith("#REDIRECT") or raw_text.strip().upper().startswith("#REDIRECCIÓN"):
            skipped_redirect += 1
            continue

        # Strip wikitext markup
        text = strip_wikitext(raw_text)

        # Optional deeper cleaning with wikitextparser
        if has_wtp:
            try:
                parsed = wtp.parse(text)
                text = parsed.plain_text()
            except Exception:
                pass  # Fall back to regex-cleaned version

        # Apply Guarani normalization
        text = normalize(text)

        # Filter stubs
        if len(text) < MIN_ARTICLE_LENGTH:
            skipped_short += 1
            continue

        articles.append({"text": text})
        processed += 1

        if processed % 500 == 0:
            print(f"  Procesados: {processed} articulos ...")

    print(f"\nResultados:")
    print(f"  Articulos validos: {processed}")
    print(f"  Redirects saltados: {skipped_redirect}")
    print(f"  Stubs saltados (<{MIN_ARTICLE_LENGTH} chars): {skipped_short}")

    return articles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia el dump XML de Wikipedia Guarani para entrenamiento LM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Dump XML (bz2) de entrada (default: {DEFAULT_INPUT}).",
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
        default=MIN_ARTICLE_LENGTH,
        help=f"Longitud minima de articulo en chars (default: {MIN_ARTICLE_LENGTH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MIN_ARTICLE_LENGTH
    MIN_ARTICLE_LENGTH = args.min_length

    if not args.input.exists():
        print(f"[error] Archivo de entrada no encontrado: {args.input}")
        print("  Ejecuta primero: python scripts/download_data.py --sources gnwiki")
        sys.exit(1)

    articles = parse_dump(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for article in articles:
            fh.write(json.dumps(article, ensure_ascii=False) + "\n")

    size_mb = args.output.stat().st_size / (1 << 20)
    print(f"\nGuardado: {args.output} ({len(articles)} articulos, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

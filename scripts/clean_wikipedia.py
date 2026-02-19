"""Clean and deduplicate Guarani Wikipedia from XML dump + HuggingFace JSONL.

Processes BOTH sources:
  1. XML dump (bz2): data/raw/gnwiki-latest-pages-articles.xml.bz2
  2. HF JSONL:       data/raw/wikipedia_hf/wikipedia_gn.jsonl

Cleans wiki markup, applies Guarani normalization, deduplicates across sources,
filters short articles, and outputs a single clean JSONL file.

Output: data/processed/wikipedia_clean.jsonl
        Format: {"text": "...", "source": "wikipedia", "id": "wiki_0001"}
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_XML_INPUT = PROJECT_ROOT / "data" / "raw" / "gnwiki-latest-pages-articles.xml.bz2"
DEFAULT_HF_INPUT = PROJECT_ROOT / "data" / "raw" / "wikipedia_hf" / "wikipedia_gn.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "wikipedia_clean.jsonl"

# Minimum word count after cleaning to keep an article
MIN_WORDS = 50

# MediaWiki XML namespace
MW_NS = "http://www.mediawiki.org/xml/export-0.10/"

# ---------------------------------------------------------------------------
# Wikitext cleaning (regex-based)
# ---------------------------------------------------------------------------

_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
_RE_REF_SELF = re.compile(r"<ref[^/]*/\s*>")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_RE_CATEGORY = re.compile(
    r"\[\[(?:Category|Categor[ií]a|Ñemohenda):[^\]]*\]\]", re.IGNORECASE
)
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
_RE_REDIRECT = re.compile(r"^#REDIRECT", re.IGNORECASE)
_RE_DISAMBIG = re.compile(
    r"\{\{(?:desambiguaci[oó]n|disambiguation|desambig|ñemomba'e|set)\s*\}\}",
    re.IGNORECASE,
)


def strip_wikitext(wikitext: str) -> str:
    """Remove MediaWiki markup from article wikitext, returning plain text."""
    text = wikitext

    # HTML comments
    text = _RE_COMMENT.sub("", text)

    # References
    text = _RE_REF.sub("", text)
    text = _RE_REF_SELF.sub("", text)

    # Tables
    text = _RE_TABLE.sub("", text)

    # Templates (nested -- iterate until stable)
    prev = None
    while prev != text:
        prev = text
        text = _RE_TEMPLATE.sub("", text)

    # Files/images
    text = _RE_FILE.sub("", text)

    # Categories
    text = _RE_CATEGORY.sub("", text)

    # Interwiki links
    text = _RE_INTERWIKI.sub("", text)

    # Piped links [[target|display]] -> display
    text = _RE_LINK_PIPED.sub(r"\1", text)

    # Simple links [[target]] -> target
    text = _RE_LINK_SIMPLE.sub(r"\1", text)

    # External links with text [url text] -> text
    text = _RE_EXTERNAL_LINK_TEXT.sub(r"\1", text)

    # Bare external links [url] -> ""
    text = _RE_EXTERNAL_LINK_BARE.sub("", text)

    # Remaining HTML tags
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


def clean_wikitext_wtp(text: str) -> str:
    """Use wikitextparser for deeper cleanup after regex pass."""
    try:
        import wikitextparser as wtp

        parsed = wtp.parse(text)
        return parsed.plain_text()
    except ImportError:
        return text
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Deduplication fingerprint
# ---------------------------------------------------------------------------


def _fingerprint(text: str) -> str:
    """Create a dedup fingerprint from the first 50 chars of normalized lowercase text."""
    # Use first 50 non-whitespace chars for overlap detection
    condensed = " ".join(text.lower().split())[:200]
    return hashlib.md5(condensed.encode("utf-8")).hexdigest()


def _full_hash(text: str) -> str:
    """Full content hash for exact dedup."""
    condensed = " ".join(text.lower().split())
    return hashlib.md5(condensed.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Source 1: XML dump parsing (bz2 + xml.etree.ElementTree)
# ---------------------------------------------------------------------------


def _tag(local: str) -> str:
    """Build a namespaced tag. Tries common MW namespaces."""
    return f"{{{MW_NS}}}{local}"


def parse_xml_dump(xml_path: Path) -> list[dict[str, str]]:
    """Parse MediaWiki XML dump using stdlib xml.etree.ElementTree.

    Iteratively parses the XML to avoid loading the entire tree into memory.
    Extracts articles from namespace 0, skips redirects and disambiguation pages.
    """
    articles: list[dict[str, str]] = []
    skipped_redirect = 0
    skipped_disambig = 0
    skipped_short = 0
    processed = 0

    print(f"[XML] Parseando dump: {xml_path}")

    # Detect the actual namespace from the XML
    ns = None
    opener = bz2.open(xml_path, "rt", encoding="utf-8") if xml_path.suffix == ".bz2" else open(
        xml_path, "r", encoding="utf-8"
    )

    with opener as f:
        for event, elem in ET.iterparse(f, events=("start", "end")):
            # Detect namespace from the first element
            if ns is None and event == "start":
                tag = elem.tag
                if tag.startswith("{"):
                    ns = tag[1:tag.index("}")]
                else:
                    ns = ""
                break

    if ns is None:
        ns = MW_NS

    def nstag(local: str) -> str:
        return f"{{{ns}}}{local}" if ns else local

    # Second pass: parse pages
    opener = bz2.open(xml_path, "rt", encoding="utf-8") if xml_path.suffix == ".bz2" else open(
        xml_path, "r", encoding="utf-8"
    )

    with opener as f:
        page_tag = nstag("page")
        title_tag = nstag("title")
        ns_tag = nstag("ns")
        redirect_tag = nstag("redirect")
        revision_tag = nstag("revision")
        text_tag = nstag("text")

        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != page_tag:
                continue

            # Only process namespace 0 (articles)
            ns_elem = elem.find(ns_tag)
            if ns_elem is not None and ns_elem.text != "0":
                elem.clear()
                continue

            # Skip redirects (element-based detection)
            if elem.find(redirect_tag) is not None:
                skipped_redirect += 1
                elem.clear()
                continue

            # Get title
            title_elem = elem.find(title_tag)
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            # Get the text of the latest revision
            revision_elem = elem.find(revision_tag)
            if revision_elem is None:
                elem.clear()
                continue

            text_elem = revision_elem.find(text_tag)
            if text_elem is None or not text_elem.text:
                elem.clear()
                continue

            raw_text = text_elem.text

            # Skip text-based redirects
            if _RE_REDIRECT.match(raw_text.strip()):
                skipped_redirect += 1
                elem.clear()
                continue

            # Skip disambiguation pages
            if _RE_DISAMBIG.search(raw_text):
                skipped_disambig += 1
                elem.clear()
                continue

            # Clean wiki markup (regex first, then wikitextparser)
            text = strip_wikitext(raw_text)
            text = clean_wikitext_wtp(text)

            # Apply Guarani normalization
            text = normalize(text)

            # Filter short articles
            word_count = len(text.split())
            if word_count < MIN_WORDS:
                skipped_short += 1
                elem.clear()
                continue

            articles.append({"title": title, "text": text})
            processed += 1

            if processed % 500 == 0:
                print(f"  [XML] Procesados: {processed} articulos ...")

            # Free memory
            elem.clear()

    print(f"  [XML] Resultados:")
    print(f"    Articulos validos: {processed}")
    print(f"    Redirects saltados: {skipped_redirect}")
    print(f"    Disambig saltados: {skipped_disambig}")
    print(f"    Stubs saltados (<{MIN_WORDS} palabras): {skipped_short}")

    return articles


# ---------------------------------------------------------------------------
# Source 2: HuggingFace JSONL
# ---------------------------------------------------------------------------


def parse_hf_jsonl(jsonl_path: Path) -> list[dict[str, str]]:
    """Parse HuggingFace Wikipedia JSONL (pre-extracted plain text).

    Each line is {"text": "...", ...}. Text is already mostly clean but may
    still contain some wiki artifacts. We apply the same cleaning + normalization.
    """
    articles: list[dict[str, str]] = []
    skipped_short = 0
    skipped_empty = 0
    processed = 0

    print(f"[HF] Parseando JSONL: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [HF] Linea {line_num}: JSON invalido, saltando.")
                continue

            text = record.get("text", "")
            if not text:
                skipped_empty += 1
                continue

            title = record.get("title", "")

            # The HF version is pre-extracted text but may still have some markup
            # Apply light wiki cleanup just in case
            text = strip_wikitext(text)

            # Apply Guarani normalization
            text = normalize(text)

            # Filter short articles
            word_count = len(text.split())
            if word_count < MIN_WORDS:
                skipped_short += 1
                continue

            articles.append({"title": title, "text": text})
            processed += 1

            if processed % 1000 == 0:
                print(f"  [HF] Procesados: {processed} articulos ...")

    print(f"  [HF] Resultados:")
    print(f"    Articulos validos: {processed}")
    print(f"    Vacios saltados: {skipped_empty}")
    print(f"    Stubs saltados (<{MIN_WORDS} palabras): {skipped_short}")

    return articles


# ---------------------------------------------------------------------------
# Deduplication & merge
# ---------------------------------------------------------------------------


def deduplicate(
    xml_articles: list[dict[str, str]],
    hf_articles: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge and deduplicate articles from both sources.

    Priority: XML dump (richer text) over HF JSONL.
    Dedup strategy:
      1. Full content hash (exact duplicates)
      2. Fingerprint from first ~200 chars (near-duplicates / same article different extraction)
    """
    seen_hashes: set[str] = set()
    seen_fingerprints: set[str] = set()
    merged: list[dict[str, str]] = []
    dup_count = 0

    # Add XML articles first (they have richer text from full markup parsing)
    for article in xml_articles:
        text = article["text"]
        fh = _full_hash(text)
        fp = _fingerprint(text)

        if fh in seen_hashes or fp in seen_fingerprints:
            dup_count += 1
            continue

        seen_hashes.add(fh)
        seen_fingerprints.add(fp)
        merged.append(article)

    # Add HF articles, skip duplicates
    hf_new = 0
    hf_dup = 0
    for article in hf_articles:
        text = article["text"]
        fh = _full_hash(text)
        fp = _fingerprint(text)

        if fh in seen_hashes or fp in seen_fingerprints:
            hf_dup += 1
            continue

        seen_hashes.add(fh)
        seen_fingerprints.add(fp)
        merged.append(article)
        hf_new += 1

    print(f"\n[Dedup] Resultados:")
    print(f"  XML articulos unicos: {len(xml_articles) - dup_count}")
    print(f"  XML duplicados internos: {dup_count}")
    print(f"  HF articulos nuevos (no en XML): {hf_new}")
    print(f"  HF duplicados (ya en XML): {hf_dup}")
    print(f"  Total despues de dedup: {len(merged)}")

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia y deduplica Wikipedia Guarani (XML dump + HF JSONL)."
    )
    parser.add_argument(
        "--xml-input",
        type=Path,
        default=DEFAULT_XML_INPUT,
        help=f"Dump XML (bz2) de Wikipedia (default: {DEFAULT_XML_INPUT.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--hf-input",
        type=Path,
        default=DEFAULT_HF_INPUT,
        help=f"JSONL de HuggingFace (default: {DEFAULT_HF_INPUT.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Archivo JSONL de salida (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORDS,
        help=f"Minimo de palabras por articulo (default: {MIN_WORDS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MIN_WORDS
    MIN_WORDS = args.min_words

    print("=" * 60)
    print("GuaraniLM - Limpieza Wikipedia Guarani")
    print(f"  Filtro: min_words={MIN_WORDS}")
    print("=" * 60)

    # --- Source 1: XML dump ---
    xml_articles: list[dict[str, str]] = []
    if args.xml_input.exists():
        xml_articles = parse_xml_dump(args.xml_input)
    else:
        print(f"\n[XML] Archivo no encontrado: {args.xml_input}")
        print("  Saltando fuente XML. Ejecuta: python scripts/download_data.py --sources gnwiki")

    # --- Source 2: HuggingFace JSONL ---
    hf_articles: list[dict[str, str]] = []
    if args.hf_input.exists():
        hf_articles = parse_hf_jsonl(args.hf_input)
    else:
        print(f"\n[HF] Archivo no encontrado: {args.hf_input}")
        print("  Saltando fuente HF.")

    if not xml_articles and not hf_articles:
        print("\n[error] No se encontro ninguna fuente de datos. Abortando.")
        sys.exit(1)

    # --- Deduplicate & merge ---
    merged = deduplicate(xml_articles, hf_articles)

    # --- Write output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_words = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, article in enumerate(merged):
            record = {
                "text": article["text"],
                "source": "wikipedia",
                "id": f"wiki_{i:05d}",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_words += len(article["text"].split())

    size_mb = args.output.stat().st_size / (1 << 20)

    # --- Stats ---
    total_input = len(xml_articles) + len(hf_articles)
    duplicates_removed = total_input - len(merged)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"  Fuente XML:          {len(xml_articles)} articulos")
    print(f"  Fuente HF:           {len(hf_articles)} articulos")
    print(f"  Total procesados:    {total_input}")
    print(f"  Duplicados removidos:{duplicates_removed}")
    print(f"  Articulos finales:   {len(merged)}")
    print(f"  Palabras estimadas:  {total_words:,}")
    print(f"  Archivo:             {args.output} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()

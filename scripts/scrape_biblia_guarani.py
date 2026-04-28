"""
Scraper for the Guaraní Bible from bible.com (YouVersion).

Downloads Guaraní Paraguayan Bible versions:
  - Tûpâ Ñandeyára 1913 (GRN1913)
  - Ñandejara Ñe'ẽ (GDC2006)
  - Ñandejára Ñe'ẽ (GPC2006)

Uses the YouVersion web interface to extract chapter text.
"""
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "biblia_guarani"

# YouVersion Bible versions for Guaraní Paraguayo
# Verified working versions (server-rendered content)
VERSIONS = {
    "GPC2006": {
        "id": 66,
        "name": "Ñandejára Ñe'ẽ (2006)",
        "lang": "gug",
    },
    "NNP2015": {
        "id": 3146,
        "name": "Ñanderuete Ñe'ẽ Porã (2015)",
        "lang": "gug",
    },
    "GDC2006": {
        "id": 3374,
        "name": "Ñandejara Ñe'ẽ ✟ (2006)",
        "lang": "gug",
    },
}

# Standard Bible books (abbreviation, full name, chapters)
BIBLE_BOOKS = [
    ("GEN", "Génesis", 50), ("EXO", "Éxodo", 40), ("LEV", "Levítico", 27),
    ("NUM", "Números", 36), ("DEU", "Deuteronomio", 34), ("JOS", "Josué", 24),
    ("JDG", "Jueces", 21), ("RUT", "Rut", 4), ("1SA", "1 Samuel", 31),
    ("2SA", "2 Samuel", 24), ("1KI", "1 Reyes", 22), ("2KI", "2 Reyes", 25),
    ("1CH", "1 Crónicas", 29), ("2CH", "2 Crónicas", 36), ("EZR", "Esdras", 10),
    ("NEH", "Nehemías", 13), ("EST", "Ester", 10), ("JOB", "Job", 42),
    ("PSA", "Salmos", 150), ("PRO", "Proverbios", 31), ("ECC", "Eclesiastés", 12),
    ("SNG", "Cantar de los Cantares", 8), ("ISA", "Isaías", 66),
    ("JER", "Jeremías", 52), ("LAM", "Lamentaciones", 5), ("EZK", "Ezequiel", 48),
    ("DAN", "Daniel", 12), ("HOS", "Oseas", 14), ("JOL", "Joel", 3),
    ("AMO", "Amós", 9), ("OBA", "Abdías", 1), ("JON", "Jonás", 4),
    ("MIC", "Miqueas", 7), ("NAM", "Nahúm", 3), ("HAB", "Habacuc", 3),
    ("ZEP", "Sofonías", 3), ("HAG", "Hageo", 2), ("ZEC", "Zacarías", 14),
    ("MAL", "Malaquías", 4),
    # New Testament
    ("MAT", "Mateo", 28), ("MRK", "Marcos", 16), ("LUK", "Lucas", 24),
    ("JHN", "Juan", 21), ("ACT", "Hechos", 28), ("ROM", "Romanos", 16),
    ("1CO", "1 Corintios", 16), ("2CO", "2 Corintios", 13), ("GAL", "Gálatas", 6),
    ("EPH", "Efesios", 6), ("PHP", "Filipenses", 4), ("COL", "Colosenses", 4),
    ("1TH", "1 Tesalonicenses", 5), ("2TH", "2 Tesalonicenses", 3),
    ("1TI", "1 Timoteo", 6), ("2TI", "2 Timoteo", 4), ("TIT", "Tito", 3),
    ("PHM", "Filemón", 1), ("HEB", "Hebreos", 13), ("JAS", "Santiago", 5),
    ("1PE", "1 Pedro", 5), ("2PE", "2 Pedro", 3), ("1JN", "1 Juan", 5),
    ("2JN", "2 Juan", 1), ("3JN", "3 Juan", 1), ("JUD", "Judas", 1),
    ("REV", "Apocalipsis", 22),
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "gn,es;q=0.9",
})


def fetch_chapter_text(version_id, book_abbr, chapter):
    """Fetch a single chapter from YouVersion."""
    url = f"https://www.bible.com/bible/{version_id}/{book_abbr}.{chapter}"

    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception as e:
        print(f"      ERROR fetching {book_abbr}.{chapter}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Check if chapter is available
    main = soup.select_one("main")
    if not main:
        return None

    main_text = main.get_text(" ", strip=True)
    if "not available" in main_text.lower():
        return None

    # Remove navigation/UI elements from main
    for tag in main.select("button, nav, [class*='nav'], [class*='toolbar'], "
                           "[class*='footer'], [class*='header'], script, style"):
        tag.decompose()

    # Extract clean text
    text = main.get_text("\n", strip=True)

    # Remove common UI strings
    ui_strings = [
        "Parallel", "READER SETTINGS", "YouVersion",
        "Encouraging and challenging", "Ministry", "About",
        "Careers", "Volunteer", "Blog", "Press", "Useful Links",
        "Help", "Donate", "Bible Versions", "Audio Bibles",
        "Bible Languages", "Verse of the Day", "A Digital Ministry",
        "Life.Church", "Privacy Policy", "Terms",
        "Vulnerability Disclosure", "Facebook", "Twitter", "Instagram",
    ]
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(ui in line for ui in ui_strings):
            continue
        if line.startswith("©"):
            continue
        # Skip very short lines that are likely UI (POI, etc)
        if len(line) < 3 and not line[0].isdigit():
            continue
        clean_lines.append(line)

    # Remove the book name + chapter header (first line usually)
    if clean_lines and re.match(r'^[A-ZÑ].*\d+\s+\w{3,}', clean_lines[0]):
        clean_lines = clean_lines[1:]

    text = "\n".join(clean_lines)

    if len(text) < 50:
        return None

    return text


def scrape_version(version_key, version_info, books_to_scrape=None):
    """Scrape all chapters of a Bible version."""
    version_id = version_info["id"]
    version_name = version_info["name"]

    version_dir = OUTPUT_DIR / version_key
    version_dir.mkdir(parents=True, exist_ok=True)

    all_texts = []
    books = books_to_scrape or BIBLE_BOOKS

    for book_abbr, book_name, num_chapters in books:
        book_texts = []
        print(f"  {book_name} ({book_abbr}): ", end="", flush=True)

        for chapter in range(1, num_chapters + 1):
            cache_file = version_dir / f"{book_abbr}_{chapter}.txt"

            if cache_file.exists():
                text = cache_file.read_text(encoding="utf-8")
                print(".", end="", flush=True)
            else:
                text = fetch_chapter_text(version_id, book_abbr, chapter)
                if text:
                    cache_file.write_text(text, encoding="utf-8")
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
                time.sleep(1.5)  # Be respectful with rate limiting

            if text:
                book_texts.append({
                    "book": book_abbr,
                    "book_name": book_name,
                    "chapter": chapter,
                    "text": text,
                })

        print(f" ({len(book_texts)}/{num_chapters} chapters)")

        if book_texts:
            all_texts.extend(book_texts)

    return all_texts


def main():
    parser = argparse.ArgumentParser(description="Scrape Guaraní Bible from YouVersion")
    parser.add_argument("--version", choices=list(VERSIONS.keys()),
                        default="NNP2015", help="Bible version to scrape")
    parser.add_argument("--nt-only", action="store_true",
                        help="Only scrape New Testament")
    parser.add_argument("--ot-only", action="store_true",
                        help="Only scrape Old Testament")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    version_key = args.version
    version_info = VERSIONS[version_key]

    books = BIBLE_BOOKS
    if args.nt_only:
        books = [b for b in BIBLE_BOOKS if BIBLE_BOOKS.index(b) >= 39]
    elif args.ot_only:
        books = [b for b in BIBLE_BOOKS if BIBLE_BOOKS.index(b) < 39]

    print("=" * 60)
    print(f"Scraping Guaraní Bible: {version_info['name']}")
    print(f"Version ID: {version_info['id']}")
    print(f"Books to scrape: {len(books)}")
    print("=" * 60)

    all_texts = scrape_version(version_key, version_info, books)

    # Save combined output
    output_file = OUTPUT_DIR / f"biblia_{version_key}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_texts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Also save as plain text
    plain_file = OUTPUT_DIR / f"biblia_{version_key}.txt"
    with open(plain_file, "w", encoding="utf-8") as f:
        for item in all_texts:
            f.write(f"\n--- {item['book_name']} {item['chapter']} ---\n")
            f.write(item["text"] + "\n")

    total_chars = sum(len(t["text"]) for t in all_texts)
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Chapters scraped: {len(all_texts)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  JSONL: {output_file}")
    print(f"  Plain text: {plain_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

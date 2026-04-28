"""
Scraper for JW Watchtower Online Library in Guaraní (wol.jw.org).

Crawls all 12 publication categories, discovers chapter/article doc IDs,
and extracts clean Guaraní text content.

URL patterns:
  - Categories: /gug/wol/library/r48/lp-gi/puvlikasionkuéra/{category}
  - Publications: /gug/wol/publication/r48/lp-gi/{code}
  - Documents: /gug/wol/d/r48/lp-gi/{docid}

Usage:
  python scripts/scrape_wol_jw.py                  # Scrape all categories
  python scripts/scrape_wol_jw.py --category lívro  # Scrape only books
  python scripts/scrape_wol_jw.py --max-docs 100    # Limit documents
"""
import json
import re
import time
import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "wol_jw"

BASE_URL = "https://wol.jw.org"
LANG_PATH = "/gug/wol"
LIBRARY_PATH = f"{LANG_PATH}/library/r48/lp-gi/puvlikasionkuéra"
PUB_PATH = f"{LANG_PATH}/publication/r48/lp-gi"
DOC_PATH = f"{LANG_PATH}/d/r48/lp-gi"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "gn,es;q=0.9",
})

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests


def fetch_page(url):
    """Fetch a page and return BeautifulSoup object."""
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"    ERROR fetching {url}: {e}")
        return None


def get_categories():
    """Get all publication categories from the library page."""
    url = f"{BASE_URL}{LIBRARY_PATH}"
    soup = fetch_page(url)
    if not soup:
        return []

    categories = []
    seen_slugs = set()
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name or len(name) < 2:
            continue

        # Match links like /gug/wol/library/r48/lp-gi/puvlikasionkuéra/{slug}
        # where {slug} is a single path segment (no further nesting)
        match = re.search(r'/puvlikasionku[^\s/]+/([^/]+)$', href.rstrip("/"))
        if match:
            slug = match.group(1)
            # Skip the self-link to the publications page itself
            if slug.startswith("puvlikasion"):
                continue
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                full_url = urljoin(BASE_URL, href)
                categories.append((slug, name, full_url))

    return categories


def extract_doc_ids_from_page(soup):
    """Extract all document IDs from links on a page."""
    doc_ids = []
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        # Match /gug/wol/d/r48/lp-gi/{docid}
        match = re.search(r'/d/r48/lp-gi/(\d+)', href)
        if match:
            doc_id = match.group(1)
            title = link.get_text(strip=True)
            if doc_id not in [d[0] for d in doc_ids]:
                doc_ids.append((doc_id, title))
    return doc_ids


def extract_sub_links(soup):
    """Extract publication and library sub-links from a page."""
    pub_links = []
    lib_links = []

    for link in soup.select("a[href]"):
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name or len(name) < 2:
            continue

        full_url = urljoin(BASE_URL, href)

        # Publication links: /gug/wol/publication/r48/lp-gi/{code}
        if "/publication/r48/lp-gi/" in href:
            if full_url not in [p[1] for p in pub_links]:
                pub_links.append((name, full_url))

        # Library sub-links (deeper than category level)
        elif "/puvlikasionkuéra/" in href:
            parts = href.rstrip("/").split("/")
            try:
                idx = parts.index("puvlikasionkuéra")
                # Only follow links that are deeper than category level
                if idx + 2 < len(parts):
                    if full_url not in [l[1] for l in lib_links]:
                        lib_links.append((name, full_url))
            except ValueError:
                continue

    return pub_links, lib_links


def discover_doc_ids(category_url, category_name, max_depth=3):
    """Recursively discover all document IDs in a category."""
    all_doc_ids = []
    visited = set()

    def _crawl(url, depth=0):
        if depth > max_depth or url in visited:
            return
        visited.add(url)

        indent = "    " * (depth + 1)
        # Show which page we're crawling
        short_url = url.split("puvlikasionkuéra/")[-1] if "puvlikasionkuéra/" in url else url.split("/")[-1]
        print(f"{indent}Crawling: {short_url} (depth={depth}, visited={len(visited)}, docs={len(all_doc_ids)})", flush=True)

        soup = fetch_page(url)
        if not soup:
            return
        time.sleep(REQUEST_DELAY)

        # Extract doc IDs directly on this page
        doc_ids = extract_doc_ids_from_page(soup)
        for doc_id, title in doc_ids:
            if doc_id not in [d[0] for d in all_doc_ids]:
                all_doc_ids.append((doc_id, title))

        # Find sub-pages to crawl
        pub_links, lib_links = extract_sub_links(soup)

        # Crawl publication pages (they contain chapter listings with doc IDs)
        for name, pub_url in pub_links:
            if pub_url not in visited:
                visited.add(pub_url)
                pub_soup = fetch_page(pub_url)
                if pub_soup:
                    pub_doc_ids = extract_doc_ids_from_page(pub_soup)
                    for doc_id, title in pub_doc_ids:
                        if doc_id not in [d[0] for d in all_doc_ids]:
                            all_doc_ids.append((doc_id, title))
                    if pub_doc_ids:
                        print(f"{indent}  {name}: {len(pub_doc_ids)} docs", flush=True)
                time.sleep(REQUEST_DELAY)

        # Crawl library sub-pages
        print(f"{indent}  Sub-pages to crawl: {len(lib_links)}", flush=True)
        for name, lib_url in lib_links:
            _crawl(lib_url, depth + 1)

    _crawl(category_url)
    return all_doc_ids


def extract_document_text(doc_id):
    """Fetch and extract clean text from a document page."""
    url = f"{BASE_URL}{DOC_PATH}/{doc_id}"
    soup = fetch_page(url)
    if not soup:
        return None, None

    # Get the article/document title
    title = ""
    title_elem = soup.select_one("h1, h2, .articleTitle, header h1")
    if title_elem:
        title = title_elem.get_text(strip=True)

    # Find the main content area
    # WOL uses article element or specific content divs
    content_elem = soup.select_one(
        "article, .docContent, .bodyTxt, "
        "#article, .content-area, main"
    )

    if not content_elem:
        # Fall back to the entire body minus nav/footer
        content_elem = soup.select_one("body")
        if not content_elem:
            return title, None

    # Remove navigation, footer, UI elements
    for tag in content_elem.select(
        "nav, footer, header, button, script, style, "
        "[class*='nav'], [class*='toolbar'], [class*='footer'], "
        "[class*='header'], [class*='menu'], [class*='sidebar'], "
        ".groupFootnote, .footnoteLink"
    ):
        tag.decompose()

    # Extract text
    text = content_elem.get_text("\n", strip=True)

    # Clean up UI strings
    ui_strings = [
        "BIBLIA", "PUVLIKASIONKUÉRA", "RREUNIONKUÉRA",
        "wol.jw.org", "jw.org", "Watchtower", "ONLINE LIBRARY",
        "Copyright", "Terms of Use", "Privacy Policy",
        "Log In", "Settings", "Share", "Feedback",
    ]
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(ui.lower() == line.lower() for ui in ui_strings):
            continue
        if line.startswith("©"):
            continue
        # Skip very short navigation-like lines
        if len(line) < 3:
            continue
        clean_lines.append(line)

    text = "\n".join(clean_lines)

    if len(text) < 50:
        return title, None

    return title, text


def scrape_category(category_slug, category_name, category_url, max_docs=None):
    """Scrape all documents in a category."""
    print(f"\n{'=' * 60}")
    print(f"Category: {category_name} ({category_slug})")
    print(f"URL: {category_url}")
    print(f"{'=' * 60}")

    cat_dir = OUTPUT_DIR / category_slug
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Discover all document IDs
    print("  Discovering documents...")
    doc_ids = discover_doc_ids(category_url, category_name)
    print(f"  Found {len(doc_ids)} documents")

    if max_docs:
        doc_ids = doc_ids[:max_docs]
        print(f"  Limited to {max_docs} documents")

    # Fetch each document
    documents = []
    for i, (doc_id, doc_title) in enumerate(doc_ids):
        cache_file = cat_dir / f"doc_{doc_id}.json"

        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                documents.append(json.load(f))
            continue

        print(f"  [{i+1}/{len(doc_ids)}] {doc_title[:60]}...", end=" ", flush=True)

        title, text = extract_document_text(doc_id)
        time.sleep(REQUEST_DELAY)

        if text and len(text.strip()) > 50:
            record = {
                "source": "wol_jw",
                "category": category_slug,
                "doc_id": doc_id,
                "title": title or doc_title,
                "url": f"{BASE_URL}{DOC_PATH}/{doc_id}",
                "text": text.strip(),
                "chars": len(text.strip()),
            }
            documents.append(record)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"OK ({len(text.strip())} chars)")
        else:
            print("EMPTY")

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Scrape JW Watchtower Online Library in Guaraní"
    )
    parser.add_argument("--category", type=str, default=None,
                        help="Specific category slug to scrape")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max documents per category")
    parser.add_argument("--list-categories", action="store_true",
                        help="List available categories and exit")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching categories...")
    categories = get_categories()

    if not categories:
        print("ERROR: Could not fetch categories. Check network connection.")
        return

    if args.list_categories:
        print(f"\nAvailable categories ({len(categories)}):")
        for slug, name, url in categories:
            print(f"  {slug}: {name}")
        return

    # Filter to specific category if requested
    if args.category:
        categories = [(s, n, u) for s, n, u in categories if s == args.category]
        if not categories:
            print(f"ERROR: Category '{args.category}' not found")
            return

    all_documents = []

    for slug, name, url in categories:
        docs = scrape_category(slug, name, url, args.max_docs)
        all_documents.extend(docs)

    # Save combined output
    output_file = OUTPUT_DIR / "all_wol_jw.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Summary
    total_chars = sum(d.get("chars", 0) for d in all_documents)
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    for slug, name, _ in categories:
        count = sum(1 for d in all_documents if d.get("category") == slug)
        chars = sum(d.get("chars", 0) for d in all_documents if d.get("category") == slug)
        if count > 0:
            print(f"  {name}: {count} docs, {chars:,} chars")
    print(f"  Total documents: {len(all_documents)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  Output: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

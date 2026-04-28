"""
Scraper for Agencia IP (ip.gov.py) - "En Guaraní" section.

Paraguay's government news agency publishes news articles in Guaraní.
~420 articles across 35 pages.

URL patterns:
  - Listing: /ip/en-guarani/page/{N}/
  - Articles: /ip/YYYY/MM/DD/article-slug/
  - Content in: .td-post-content

Usage:
  python scripts/scrape_ip_gov_guarani.py              # Scrape all
  python scripts/scrape_ip_gov_guarani.py --max-pages 5 # First 5 pages
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
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "ip_gov_guarani"

BASE_URL = "https://www.ip.gov.py"
LISTING_URL = f"{BASE_URL}/ip/en-guarani/"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "gn,es;q=0.9",
})

REQUEST_DELAY = 1.0


def get_article_urls(max_pages=50):
    """Collect article URLs from listing pages."""
    article_urls = []
    seen = set()

    for page in range(1, max_pages + 1):
        if page == 1:
            url = LISTING_URL
        else:
            url = f"{LISTING_URL}page/{page}/"

        print(f"  Listing page {page}...", end=" ", flush=True)
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                print("404 - stopping")
                break
            resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        new_count = 0
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            # Match article URLs: /ip/YYYY/MM/DD/slug/
            if re.search(r'/ip/\d{4}/\d{2}/\d{2}/[^/]+/?$', href):
                full_url = urljoin(BASE_URL, href)
                if full_url not in seen:
                    seen.add(full_url)
                    article_urls.append(full_url)
                    new_count += 1

        print(f"{new_count} new articles (total: {len(article_urls)})")

        if new_count == 0:
            print("  No new articles found. Stopping.")
            break

        time.sleep(REQUEST_DELAY)

    return article_urls


def scrape_article(url):
    """Fetch and extract text from a single article."""
    try:
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return None, None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_elem = soup.select_one("h1.entry-title, h1.tdb-title-text, h1")
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Extract article body (.td-post-content is the WordPress Newspaper theme class)
    body_elem = soup.select_one(
        ".td-post-content, .tdb-block-inner .tdb-block-inner, "
        "article .entry-content, .post-content"
    )
    if not body_elem:
        return title, None

    # Remove unwanted elements
    for tag in body_elem.select(
        "script, style, .sharedaddy, .jp-relatedposts, "
        "[class*='social'], [class*='share'], [class*='ad-'], "
        "figure figcaption, .wp-caption-text"
    ):
        tag.decompose()

    text = body_elem.get_text("\n", strip=True)

    # Clean up
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip short social media / sharing lines
        if len(line) < 5:
            continue
        lines.append(line)

    text = "\n".join(lines)

    if len(text) < 50:
        return title, None

    return title, text


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Agencia IP Guaraní articles"
    )
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max listing pages to crawl")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Scraping Agencia IP - En Guaraní (ip.gov.py)")
    print("=" * 60)

    # Collect article URLs
    print("\nCollecting article URLs...")
    article_urls = get_article_urls(args.max_pages)
    print(f"\nTotal articles found: {len(article_urls)}")

    # Scrape each article
    articles = []
    for i, url in enumerate(article_urls):
        # Cache check
        cache_file = OUTPUT_DIR / f"article_{i:04d}.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                articles.append(json.load(f))
            continue

        print(f"  [{i+1}/{len(article_urls)}] {url.split('/')[-2][:50]}...", end=" ", flush=True)

        title, text = scrape_article(url)
        time.sleep(REQUEST_DELAY)

        if text:
            record = {
                "source": "ip_gov_guarani",
                "url": url,
                "title": title,
                "text": text,
                "chars": len(text),
            }
            articles.append(record)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"OK ({len(text)} chars)")
        else:
            print("EMPTY")

    # Save combined output
    output_file = OUTPUT_DIR / "all_ip_gov_guarani.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    total_chars = sum(a.get("chars", 0) for a in articles)
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Total articles: {len(articles)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  Output: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

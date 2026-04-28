"""
Scraper for Guaraní news articles from Paraguayan media.

Sources:
  1. ABC Color - Remiandu section (abc.com.py/especiales/remiandu/)
     Educational articles about Guaraní language
  2. Última Hora - Guaraní section (ultimahora.com/guarani)
     News articles related to Guaraní language and culture

Both sites use dynamic loading. We scrape article links from listing pages,
then fetch each article's text content.
"""
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "news_guarani"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "gn,es;q=0.9",
})


# =============================================================================
# ABC Remiandu
# =============================================================================

def scrape_abc_remiandu(max_pages=50):
    """Scrape articles from ABC Remiandu section."""
    print("=" * 60)
    print("Scraping ABC Remiandu (abc.com.py/especiales/remiandu/)")
    print("=" * 60)

    outdir = OUTPUT_DIR / "abc_remiandu"
    outdir.mkdir(parents=True, exist_ok=True)

    # ABC Remiandu has a listing page with article links
    base_url = "https://www.abc.com.py/especiales/remiandu/"

    # Collect article URLs
    article_urls = set()

    # Try main listing page and paginated versions
    for page in range(1, max_pages + 1):
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}?page={page}"

        print(f"  Fetching listing page {page}...")
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
        except Exception as e:
            print(f"    ERROR: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find article links - ABC Remiandu uses date-based URLs
        # Pattern: /especiales/remiandu/YYYY/MM/DD/article-slug/
        links = soup.select('a[href*="remiandu"]')
        new_urls = set()
        for link in links:
            href = link.get("href", "")
            if href and re.search(r'/remiandu/\d{4}/\d{2}/\d{2}/', href):
                # Skip comment links
                if "#comments" in href:
                    continue
                full_url = urljoin("https://www.abc.com.py", href)
                if full_url not in article_urls:
                    new_urls.add(full_url)

        if not new_urls:
            print(f"    No new articles found. Stopping.")
            break

        article_urls.update(new_urls)
        print(f"    Found {len(new_urls)} new article links (total: {len(article_urls)})")
        time.sleep(1)

    print(f"\n  Total article URLs collected: {len(article_urls)}")

    # Fetch each article
    articles = []
    for i, url in enumerate(sorted(article_urls)):
        cache_file = outdir / f"article_{i:04d}.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                articles.append(json.load(f))
            continue

        print(f"  [{i+1}/{len(article_urls)}] {url}")
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title
            title_elem = soup.select_one('h1, .article-title, [class*="title"]')
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract article body
            body_elem = soup.select_one(
                'article, .article-body, .article-content, '
                '[class*="article-text"], [class*="body"]'
            )
            if body_elem:
                # Remove scripts, styles, ads
                for tag in body_elem.select('script, style, [class*="ad"], [class*="social"]'):
                    tag.decompose()
                body_text = body_elem.get_text("\n", strip=True)
            else:
                body_text = ""

            if body_text and len(body_text) > 100:
                article = {
                    "source": "abc_remiandu",
                    "url": url,
                    "title": title,
                    "text": body_text,
                    "chars": len(body_text),
                }
                articles.append(article)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(1.5)

    return articles


# =============================================================================
# Última Hora
# =============================================================================

def scrape_ultimahora_guarani(max_pages=50):
    """Scrape articles from Última Hora Guaraní section."""
    print("\n" + "=" * 60)
    print("Scraping Última Hora Guaraní (ultimahora.com/guarani)")
    print("=" * 60)

    outdir = OUTPUT_DIR / "ultimahora"
    outdir.mkdir(parents=True, exist_ok=True)

    base_url = "https://www.ultimahora.com/guarani"

    article_urls = set()

    # Fetch listing pages
    for page in range(1, max_pages + 1):
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}?page={page}"

        print(f"  Fetching listing page {page}...")
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
        except Exception as e:
            print(f"    ERROR: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        new_urls = set()
        for link in soup.select('a[href]'):
            href = link.get("href", "")
            full_url = urljoin("https://www.ultimahora.com", href)
            # Match article URLs (slug pattern)
            if (re.match(r'https://www\.ultimahora\.com/[a-z0-9-]+-n?\d*$', full_url)
                    or re.match(r'https://www\.ultimahora\.com/[a-z0-9-]{20,}$', full_url)):
                if full_url not in article_urls and full_url != base_url:
                    new_urls.add(full_url)

        if not new_urls:
            print(f"    No new articles found. Stopping.")
            break

        article_urls.update(new_urls)
        print(f"    Found {len(new_urls)} new article links (total: {len(article_urls)})")
        time.sleep(1)

    print(f"\n  Total article URLs collected: {len(article_urls)}")

    # Fetch each article
    articles = []
    for i, url in enumerate(sorted(article_urls)):
        cache_file = outdir / f"article_{i:04d}.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                articles.append(json.load(f))
            continue

        print(f"  [{i+1}/{len(article_urls)}] {url}")
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract title
            title_elem = soup.select_one('h1')
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract article body
            body_elem = soup.select_one(
                'article, [class*="article-body"], [class*="nota-cuerpo"], '
                '[class*="content-body"]'
            )
            if body_elem:
                for tag in body_elem.select('script, style, [class*="ad"], [class*="social"]'):
                    tag.decompose()
                body_text = body_elem.get_text("\n", strip=True)
            else:
                body_text = ""

            if body_text and len(body_text) > 100:
                article = {
                    "source": "ultimahora",
                    "url": url,
                    "title": title,
                    "text": body_text,
                    "chars": len(body_text),
                }
                articles.append(article)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(1.5)

    return articles


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Guaraní news articles"
    )
    parser.add_argument("--source", choices=["abc", "ultimahora", "all"],
                        default="all", help="Which source to scrape")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max listing pages to crawl per source")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_articles = []

    if args.source in ("abc", "all"):
        abc_articles = scrape_abc_remiandu(args.max_pages)
        all_articles.extend(abc_articles)

    if args.source in ("ultimahora", "all"):
        uh_articles = scrape_ultimahora_guarani(args.max_pages)
        all_articles.extend(uh_articles)

    # Save combined output
    output_file = OUTPUT_DIR / "all_news_guarani.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for article in all_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    total_chars = sum(a.get("chars", 0) for a in all_articles)
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  ABC Remiandu: {sum(1 for a in all_articles if a['source'] == 'abc_remiandu')} articles")
    print(f"  Última Hora: {sum(1 for a in all_articles if a['source'] == 'ultimahora')} articles")
    print(f"  Total articles: {len(all_articles)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  Output: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

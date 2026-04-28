"""
OCR pipeline for Oremba'e scanned PDFs.

Uses PyMuPDF to render PDF pages + Tesseract OCR for text extraction.
Requires: PyMuPDF, pytesseract, Pillow, Tesseract-OCR installed.

Usage:
  python scripts/ocr_orembae_pdfs.py                     # OCR existing PDFs
  python scripts/ocr_orembae_pdfs.py --download --max 100 # Download + OCR first 100
"""
import json
import os
import sys
import io
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "orembae"
PDF_DIR = OUTPUT_DIR / "pdfs"
TESSDATA_DIR = PROJECT_DIR / ".tessdata"

# API
API_BASE = "https://articulos.instituto.org.py/api"
ARCHIVO_URL = f"{API_BASE}/archivo"

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR.resolve())

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GuaraniLM-DataCollector/1.0 (Academic Research)"
})


def download_pdf(archivo):
    """Download a PDF from the API if not already cached."""
    dest = PDF_DIR / archivo
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = f"{ARCHIVO_URL}/{archivo}"
    try:
        resp = SESSION.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        return None


def ocr_pdf(pdf_path, lang="spa"):
    """Extract text from a scanned PDF using OCR."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []

        for page in doc:
            # Try native text extraction first (faster if available)
            native_text = page.get_text().strip()
            if native_text and len(native_text) > 50:
                text_parts.append(native_text)
                continue

            # Fall back to OCR
            mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang=lang)
            if ocr_text.strip():
                text_parts.append(ocr_text.strip())

        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        return ""


def main():
    parser = argparse.ArgumentParser(description="OCR Oremba'e PDFs")
    parser.add_argument("--download", action="store_true",
                        help="Download PDFs from API before OCR")
    parser.add_argument("--max", type=int, default=None,
                        help="Max PDFs to process")
    parser.add_argument("--lang", default="spa",
                        help="Tesseract language (default: spa)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already OCR'd files")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_file = OUTPUT_DIR / "guarani_metadata.jsonl"
    if not metadata_file.exists():
        print("ERROR: Run scrape_orembae.py first to get metadata")
        sys.exit(1)

    with open(metadata_file, encoding="utf-8") as f:
        books = [json.loads(line) for line in f]

    print(f"Loaded {len(books)} Guaraní book records")

    # Output file
    ocr_output = OUTPUT_DIR / "guarani_ocr_texts.jsonl"

    # Load already processed files
    processed = set()
    if args.skip_existing and ocr_output.exists():
        with open(ocr_output, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                processed.add(rec.get("archivo", ""))
        print(f"Already processed: {len(processed)} files")

    # Filter books to process
    to_process = []
    for book in books:
        archivo = book.get("archivo", "")
        if not archivo:
            continue
        if archivo in processed:
            continue
        to_process.append(book)

    if args.max:
        to_process = to_process[:args.max]

    print(f"PDFs to process: {len(to_process)}")

    # Process
    success = 0
    total_chars = 0

    with open(ocr_output, "a", encoding="utf-8") as fout:
        for i, book in enumerate(to_process):
            archivo = book.get("archivo", "")
            titulo = book.get("titulo", "Unknown")

            print(f"  [{i+1}/{len(to_process)}] {titulo} ({archivo})", end=" ", flush=True)

            # Download if needed
            if args.download:
                pdf_path = download_pdf(archivo)
                if not pdf_path:
                    print("DOWNLOAD FAILED")
                    continue
            else:
                pdf_path = PDF_DIR / archivo
                if not pdf_path.exists():
                    print("NOT FOUND")
                    continue

            # OCR
            text = ocr_pdf(pdf_path, lang=args.lang)

            if text and len(text.strip()) > 20:
                record = {
                    "id": book.get("id_articulo"),
                    "titulo": titulo,
                    "autor": book.get("autor", ""),
                    "anio": book.get("anio"),
                    "tipo": book.get("tipo", ""),
                    "genero": book.get("genero", ""),
                    "archivo": archivo,
                    "text": text.strip(),
                    "chars": len(text.strip()),
                    "method": "ocr",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
                total_chars += len(text.strip())
                print(f"OK ({len(text.strip())} chars)")
            else:
                print("EMPTY")

            # Rate limit downloads
            if args.download and (i + 1) % 10 == 0:
                time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Processed: {len(to_process)}")
    print(f"  Success: {success}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")
    print(f"  Output: {ocr_output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

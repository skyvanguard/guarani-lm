"""Download all raw data sources for GuaraniLM training pipeline.

Sources:
  - Wikipedia Guarani (gnwiki) XML dump from dumps.wikimedia.org
  - CulturaX Guarani subset from HuggingFace (uonlp/CulturaX)
  - Jojajovai parallel corpus from GitHub (SilasAFerreira/Jojajovai)
  - mmaguero datasets from HuggingFace (sentiment, humor, hate-speech)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GNWIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/gnwiki/latest/gnwiki-latest-pages-articles.xml.bz2"
)
JOJAJOVAI_REPO_ZIP = (
    "https://github.com/SilasAFerreira/Jojajovai/archive/refs/heads/main.zip"
)

HF_DATASETS = {
    "culturax_gn": {"path": "uonlp/CulturaX", "name": "grn", "split": "train"},
    "guarani_sentiment": {"path": "mmaguero/guarani-sentiment", "split": "train"},
    "guarani_humor": {"path": "mmaguero/guarani-humor-detection", "split": "train"},
    "guarani_hate": {"path": "mmaguero/guarani-hate-speech", "split": "train"},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, desc: str | None = None) -> None:
    """Stream-download a file with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} ya existe")
        return

    print(f"  Descargando {desc or url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, disable=total == 0
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
            pbar.update(len(chunk))


def download_gnwiki() -> None:
    """Download the latest Guarani Wikipedia articles XML dump."""
    print("\n=== Wikipedia Guarani (gnwiki) ===")
    dest = RAW_DIR / "gnwiki-latest-pages-articles.xml.bz2"
    download_file(GNWIKI_DUMP_URL, dest, desc="gnwiki dump")
    print(f"  Guardado en {dest}")


def download_culturax() -> None:
    """Download CulturaX Guarani subset via HuggingFace datasets."""
    print("\n=== CulturaX Guarani (grn) ===")
    out_dir = RAW_DIR / "culturax_gn"
    out_file = out_dir / "culturax_gn.jsonl"
    if out_file.exists():
        print(f"  [skip] {out_file.name} ya existe")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [error] 'datasets' no instalado. Ejecuta: pip install datasets")
        return

    print("  Cargando dataset desde HuggingFace (puede demorar) ...")
    cfg = HF_DATASETS["culturax_gn"]
    ds = load_dataset(
        cfg["path"],
        cfg["name"],
        split=cfg["split"],
        trust_remote_code=True,
    )

    print(f"  Guardando {len(ds)} registros en {out_file.name} ...")
    with open(out_file, "w", encoding="utf-8") as fh:
        for row in tqdm(ds, desc="CulturaX"):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Guardado en {out_file}")


def download_jojajovai() -> None:
    """Download the Jojajovai parallel corpus from GitHub."""
    print("\n=== Jojajovai Parallel Corpus ===")
    out_dir = RAW_DIR / "jojajovai"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [skip] {out_dir} ya contiene archivos")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "jojajovai.zip"
    download_file(JOJAJOVAI_REPO_ZIP, zip_path, desc="Jojajovai repo")

    print("  Extrayendo archivos ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Extract data files only (CSV, TSV, TXT, JSON)
            lower = member.lower()
            if any(lower.endswith(ext) for ext in (".csv", ".tsv", ".txt", ".json", ".jsonl")):
                # Flatten into out_dir
                filename = Path(member).name
                if filename:
                    with zf.open(member) as src, open(out_dir / filename, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    # If no specific data files found, extract everything preserving structure
    if not any(out_dir.iterdir()):
        print("  No se encontraron archivos de datos, extrayendo todo ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

    zip_path.unlink(missing_ok=True)
    print(f"  Guardado en {out_dir}")


def download_mmaguero() -> None:
    """Download mmaguero Guarani classification datasets from HuggingFace."""
    print("\n=== mmaguero Datasets ===")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [error] 'datasets' no instalado. Ejecuta: pip install datasets")
        return

    for key in ("guarani_sentiment", "guarani_humor", "guarani_hate"):
        cfg = HF_DATASETS[key]
        out_dir = RAW_DIR / "mmaguero"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{key}.jsonl"

        if out_file.exists():
            print(f"  [skip] {out_file.name} ya existe")
            continue

        print(f"  Descargando {cfg['path']} ...")
        try:
            ds = load_dataset(cfg["path"], split=cfg["split"], trust_remote_code=True)
        except Exception as exc:
            print(f"  [error] No se pudo descargar {cfg['path']}: {exc}")
            continue

        print(f"  Guardando {len(ds)} registros en {out_file.name} ...")
        with open(out_file, "w", encoding="utf-8") as fh:
            for row in tqdm(ds, desc=key):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  Guardado en {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga todas las fuentes de datos para GuaraniLM."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=["gnwiki", "culturax", "jojajovai", "mmaguero"],
        default=None,
        help="Fuentes a descargar (por defecto: todas).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio base de datos (por defecto: data/raw/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global RAW_DIR
    if args.data_dir:
        RAW_DIR = Path(args.data_dir).resolve()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sources = args.sources or ["gnwiki", "culturax", "jojajovai", "mmaguero"]

    print(f"GuaraniLM - Descarga de datos")
    print(f"Directorio destino: {RAW_DIR}")
    print(f"Fuentes: {', '.join(sources)}")

    dispatch = {
        "gnwiki": download_gnwiki,
        "culturax": download_culturax,
        "jojajovai": download_jojajovai,
        "mmaguero": download_mmaguero,
    }

    for source in sources:
        try:
            dispatch[source]()
        except Exception as exc:
            print(f"\n[ERROR] Fallo descargando {source}: {exc}")
            continue

    print("\n=== Descarga completada ===")
    print(f"Archivos en {RAW_DIR}:")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1 << 20)
            print(f"  {p.relative_to(RAW_DIR)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

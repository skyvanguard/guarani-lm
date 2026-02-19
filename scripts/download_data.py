"""Download all raw data sources for GuaraniLM training pipeline.

Confirmed sources (see docs/data_sources.md for full details):
  - Wikipedia Guarani (gnwiki) XML dump
  - HPLT 2.0 cleaned (73K docs, ~40M tokens) — largest source
  - Jojajovai parallel corpus (30K pairs, MIT)
  - AmericasNLP 2021 shared task data
  - CC-100 Guarani subset
  - Leipzig Corpora (14K sentences)
  - mmaguero datasets (sentiment, humor, offensive, emotion)
  - Alpaca-Guarani (52K instructions, low quality)
  - GUA-SPA 2023 shared task
  - NLLB-Seed (6K professional pairs)
  - FLORES-200 + Belebele (evaluation benchmarks)
  - Tatoeba sentences
"""

from __future__ import annotations

import argparse
import json
import shutil
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

CC100_GN_URL = "https://data.statmt.org/cc-100/gn.txt.xz"

JOJAJOVAI_REPO_ZIP = (
    "https://github.com/pln-fing-udelar/jojajovai/archive/refs/heads/main.zip"
)

GONGORA_REPO_ZIP = (
    "https://github.com/sgongora27/giossa-gongora-guarani-2021/archive/refs/heads/main.zip"
)

# HuggingFace datasets config: {key: {path, name?, split, subdir}}
HF_DATASETS = {
    # === Largest source: HPLT 2.0 cleaned ===
    "hplt2_cleaned": {
        "path": "HPLT/HPLT2.0_cleaned",
        "name": "grn_Latn",
        "split": "train",
        "desc": "HPLT 2.0 cleaned (73K docs, ~40M tokens)",
    },
    # === Parallel corpus ===
    "jojajovai_hf": {
        "path": "mmaguero/jojajovai",
        "split": "train",
        "desc": "Jojajovai parallel corpus (30K pairs)",
    },
    # === Classification datasets ===
    "sentiment": {
        "path": "mmaguero/gn-jopara-sentiment-analysis",
        "split": "train",
        "desc": "Guarani sentiment analysis",
    },
    "offensive": {
        "path": "mmaguero/gn-offensive-language-identification",
        "split": "train",
        "desc": "Guarani offensive language",
    },
    "humor": {
        "path": "mmaguero/gn-humor-detection",
        "split": "train",
        "desc": "Guarani humor detection",
    },
    "emotion": {
        "path": "mmaguero/gn-emotion-recognition",
        "split": "train",
        "desc": "Guarani emotion recognition",
    },
    # === Instruction dataset (low quality, Google Translate) ===
    "alpaca_guarani": {
        "path": "saillab/alpaca-guarani-cleaned",
        "split": "train",
        "desc": "Alpaca-Guarani instructions (52K, low quality MT)",
    },
    # === Shared task ===
    "gua_spa_2023": {
        "path": "mmaguero/gua-spa-2023-task-1-2",
        "split": "train",
        "desc": "GUA-SPA IberLEF 2023 code-switching",
    },
    # === Wikipedia via HuggingFace (alternative to XML dump) ===
    "wikipedia_gn": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.gn",
        "split": "train",
        "desc": "Wikipedia Guarani (HF mirror)",
    },
    # === Evaluation benchmarks ===
    "flores200": {
        "path": "facebook/flores",
        "name": "grn_Latn",
        "split": "devtest",
        "desc": "FLORES-200 Guarani eval set",
    },
    "belebele": {
        "path": "facebook/belebele",
        "name": "grn_Latn",
        "split": "test",
        "desc": "Belebele reading comprehension",
    },
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
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, disable=total == 0
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
            pbar.update(len(chunk))


def save_hf_dataset(key: str, out_dir: Path) -> None:
    """Download a HuggingFace dataset and save as JSONL."""
    cfg = HF_DATASETS[key]
    out_file = out_dir / f"{key}.jsonl"

    if out_file.exists():
        print(f"  [skip] {out_file.name} ya existe")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [error] 'datasets' no instalado. Ejecuta: pip install datasets")
        return

    print(f"  Cargando {cfg['desc']} desde HuggingFace ...")
    load_kwargs = {"path": cfg["path"], "split": cfg["split"], "trust_remote_code": True}
    if "name" in cfg:
        load_kwargs["name"] = cfg["name"]

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as exc:
        print(f"  [error] No se pudo descargar {cfg['path']}: {exc}")
        return

    count = len(ds)
    print(f"  Guardando {count:,} registros en {out_file.name} ...")
    with open(out_file, "w", encoding="utf-8") as fh:
        for row in tqdm(ds, desc=key, total=count):
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    size_mb = out_file.stat().st_size / (1 << 20)
    print(f"  Guardado: {out_file} ({size_mb:.1f} MB, {count:,} registros)")


# ---------------------------------------------------------------------------
# Source downloaders
# ---------------------------------------------------------------------------


def download_gnwiki() -> None:
    """Download the latest Guarani Wikipedia articles XML dump."""
    print("\n=== Wikipedia Guarani (gnwiki XML dump) ===")
    dest = RAW_DIR / "gnwiki-latest-pages-articles.xml.bz2"
    download_file(GNWIKI_DUMP_URL, dest, desc="gnwiki dump")


def download_hplt2() -> None:
    """Download HPLT 2.0 cleaned Guarani subset (largest source: ~40M tokens)."""
    print("\n=== HPLT 2.0 Cleaned (grn_Latn) ===")
    save_hf_dataset("hplt2_cleaned", RAW_DIR / "hplt2")


def download_jojajovai() -> None:
    """Download the Jojajovai parallel corpus from GitHub."""
    print("\n=== Jojajovai Parallel Corpus (pln-fing-udelar) ===")
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
            lower = member.lower()
            if any(lower.endswith(ext) for ext in (".csv", ".tsv", ".txt", ".json", ".jsonl")):
                filename = Path(member).name
                if filename:
                    with zf.open(member) as src, open(out_dir / filename, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    if not any(out_dir.iterdir()):
        print("  No se encontraron archivos de datos, extrayendo todo ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

    zip_path.unlink(missing_ok=True)
    print(f"  Guardado en {out_dir}")


def download_jojajovai_hf() -> None:
    """Download Jojajovai from HuggingFace (alternative to GitHub)."""
    print("\n=== Jojajovai (HuggingFace mirror) ===")
    save_hf_dataset("jojajovai_hf", RAW_DIR / "jojajovai_hf")


def download_cc100() -> None:
    """Download CC-100 Guarani subset."""
    print("\n=== CC-100 Guarani ===")
    dest = RAW_DIR / "cc100_gn.txt.xz"
    download_file(CC100_GN_URL, dest, desc="CC-100 gn")


def download_leipzig() -> None:
    """Download Leipzig Corpora Guarani (grn_community_2017)."""
    print("\n=== Leipzig Corpora (grn_community_2017) ===")
    url = "https://downloads.wortschatz-leipzig.de/corpora/grn_community_2017.tar.gz"
    dest = RAW_DIR / "grn_community_2017.tar.gz"
    download_file(url, dest, desc="Leipzig grn_community_2017")


def download_mmaguero() -> None:
    """Download mmaguero Guarani classification datasets from HuggingFace."""
    print("\n=== mmaguero Datasets ===")
    out_dir = RAW_DIR / "mmaguero"
    for key in ("sentiment", "offensive", "humor", "emotion"):
        save_hf_dataset(key, out_dir)


def download_alpaca_guarani() -> None:
    """Download Alpaca-Guarani instructions (52K, low quality Google Translate)."""
    print("\n=== Alpaca-Guarani (52K instructions) ===")
    save_hf_dataset("alpaca_guarani", RAW_DIR / "alpaca_guarani")


def download_gua_spa() -> None:
    """Download GUA-SPA 2023 IberLEF shared task data."""
    print("\n=== GUA-SPA 2023 (IberLEF) ===")
    save_hf_dataset("gua_spa_2023", RAW_DIR / "gua_spa")


def download_wikipedia_hf() -> None:
    """Download Wikipedia Guarani from HuggingFace (alternative to XML dump)."""
    print("\n=== Wikipedia Guarani (HuggingFace) ===")
    save_hf_dataset("wikipedia_gn", RAW_DIR / "wikipedia_hf")


def download_gongora() -> None:
    """Download Gongora Guarani corpus (news + tweets)."""
    print("\n=== Gongora Corpus (news + tweets) ===")
    out_dir = RAW_DIR / "gongora"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [skip] {out_dir} ya contiene archivos")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "gongora.zip"
    download_file(GONGORA_REPO_ZIP, zip_path, desc="Gongora repo")

    print("  Extrayendo archivos ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    zip_path.unlink(missing_ok=True)
    print(f"  Guardado en {out_dir}")


def download_eval_benchmarks() -> None:
    """Download evaluation benchmarks (FLORES-200, Belebele)."""
    print("\n=== Evaluation Benchmarks ===")
    out_dir = RAW_DIR / "eval"
    for key in ("flores200", "belebele"):
        save_hf_dataset(key, out_dir)


def download_tatoeba() -> None:
    """Download Tatoeba Guarani sentences."""
    print("\n=== Tatoeba Guarani ===")
    # Tatoeba exports are available via downloads page
    url = "https://downloads.tatoeba.org/exports/per_language/grn/grn_sentences.tsv.bz2"
    dest = RAW_DIR / "tatoeba_grn.tsv.bz2"
    download_file(url, dest, desc="Tatoeba grn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_SOURCES = [
    "hplt2",
    "gnwiki",
    "jojajovai",
    "jojajovai_hf",
    "cc100",
    "leipzig",
    "mmaguero",
    "alpaca_guarani",
    "gua_spa",
    "gongora",
    "tatoeba",
    "eval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga todas las fuentes de datos para GuaraniLM."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=ALL_SOURCES + ["all"],
        default=None,
        help="Fuentes a descargar (por defecto: todas). Opciones: " + ", ".join(ALL_SOURCES),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio base de datos (por defecto: data/raw/).",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Omitir fuentes grandes (HPLT 2.0) para pruebas rapidas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global RAW_DIR
    if args.data_dir:
        RAW_DIR = Path(args.data_dir).resolve()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sources = args.sources or ALL_SOURCES
    if "all" in sources:
        sources = ALL_SOURCES

    if args.skip_large and "hplt2" in sources:
        sources = [s for s in sources if s != "hplt2"]
        print("[info] Omitiendo HPLT 2.0 (--skip-large)")

    print("=" * 60)
    print("GuaraniLM - Descarga de datos")
    print("=" * 60)
    print(f"Directorio destino: {RAW_DIR}")
    print(f"Fuentes: {', '.join(sources)}")

    dispatch = {
        "hplt2": download_hplt2,
        "gnwiki": download_gnwiki,
        "jojajovai": download_jojajovai,
        "jojajovai_hf": download_jojajovai_hf,
        "cc100": download_cc100,
        "leipzig": download_leipzig,
        "mmaguero": download_mmaguero,
        "alpaca_guarani": download_alpaca_guarani,
        "gua_spa": download_gua_spa,
        "gongora": download_gongora,
        "tatoeba": download_tatoeba,
        "eval": download_eval_benchmarks,
    }

    for source in sources:
        try:
            dispatch[source]()
        except Exception as exc:
            print(f"\n[ERROR] Fallo descargando {source}: {exc}")
            continue

    print("\n" + "=" * 60)
    print("Descarga completada")
    print("=" * 60)
    print(f"\nArchivos en {RAW_DIR}:")
    total_size = 0
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1 << 20)
            total_size += size_mb
            print(f"  {p.relative_to(RAW_DIR):<50} {size_mb:>8.1f} MB")
    print(f"\n  {'TOTAL':<50} {total_size:>8.1f} MB")


if __name__ == "__main__":
    main()

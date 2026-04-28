#!/bin/bash
# =============================================================================
# GuaraniLM — Data Collection Pipeline
#
# Ejecutar desde el directorio del proyecto:
#   pip install -r scripts/requirements-scraping.txt
#   bash scripts/run_all_scrapers.sh
#
# Fuentes:
#   1. Oremba'e (orembae.org.py) — ~6,000 poemas en guaraní (PDFs)
#   2. Biblia en Guaraní (bible.com) — Texto bíblico completo
#   3. HuggingFace corpora (FineTranslations, OSCAR, HPLT)
#   4. Noticias (ABC Remiandu, Última Hora)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  GuaraniLM — Data Collection Pipeline"
echo "  Directorio: $PROJECT_DIR"
echo "============================================================"

# --- 1. Oremba'e ---
echo ""
echo "============================================================"
echo "  FUENTE 1: Oremba'e (poemas en guaraní)"
echo "  Estimado: ~30 min (descarga + extracción de PDFs)"
echo "============================================================"
echo ""

PYTHONIOENCODING=utf-8 python3 scripts/scrape_orembae.py

# --- 2. Biblia en Guaraní ---
echo ""
echo "============================================================"
echo "  FUENTE 2: Biblia en Guaraní (YouVersion)"
echo "  Estimado: ~2h (rate-limited, 1189 capítulos)"
echo "============================================================"
echo ""

PYTHONIOENCODING=utf-8 python3 scripts/scrape_biblia_guarani.py

# --- 3. HuggingFace Corpora ---
echo ""
echo "============================================================"
echo "  FUENTE 3: HuggingFace Corpora"
echo "  (FineTranslations, OSCAR, HPLT)"
echo "============================================================"
echo ""

PYTHONIOENCODING=utf-8 python3 scripts/download_hf_guarani_corpora.py

# --- 4. Noticias ---
echo ""
echo "============================================================"
echo "  FUENTE 4: Noticias en Guaraní"
echo "  (ABC Remiandu, Última Hora)"
echo "============================================================"
echo ""

PYTHONIOENCODING=utf-8 python3 scripts/scrape_news_guarani.py

# --- Summary ---
echo ""
echo "============================================================"
echo "  RECOLECCIÓN COMPLETA"
echo "============================================================"
echo ""
echo "  Datos recolectados en: data/raw/"
echo ""

# Count files and sizes
for dir in data/raw/orembae data/raw/biblia_guarani data/raw/hf_corpora data/raw/news_guarani; do
    if [ -d "$dir" ]; then
        files=$(find "$dir" -name "*.jsonl" -o -name "*.txt" -o -name "*.json" | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $dir: $files archivos ($size)"
    fi
done

echo ""
echo "  Siguiente paso:"
echo "    1. Revisar calidad de datos descargados"
echo "    2. Ejecutar scripts de preparación de datos"
echo "    3. Entrenar modelo v3 con dataset ampliado"
echo "============================================================"

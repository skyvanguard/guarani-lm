#!/bin/bash
# =============================================================================
# GuaraniLM v2 — Re-Training Pipeline (WSL2)
#
# Ejecutar desde WSL2:
#   cd /mnt/c/Users/skyva/Documents/github-contributions/guarani-lm
#   bash run_training_v2.sh
#
# Changes vs v1:
#   - CPT: 3 epochs (vs 1) for deeper Guaraní understanding
#   - SFT: 249K instructions (vs 114K) with v2 datasets, 2 epochs
#   - Estimated time: ~10-12h total on RTX 4070 8GB
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  GuaraniLM v2 — Re-Training Pipeline"
echo "  Directorio: $PROJECT_DIR"
echo "============================================================"

# --- Check GPU ---
echo ""
echo "--- Verificando GPU ---"
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi no encontrado."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# --- Setup venv ---
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creando virtualenv ---"
    python3 -m venv "$VENV_DIR"
fi

echo "--- Activando virtualenv ---"
source "$VENV_DIR/bin/activate"

# --- Install dependencies ---
echo "--- Instalando dependencias ---"
pip install --upgrade pip setuptools wheel 2>&1 | tail -1

pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -1
pip install datasets transformers accelerate peft bitsandbytes trl 2>&1 | tail -1
pip install pyyaml scikit-learn sacrebleu 2>&1 | tail -1

echo ""
echo "--- Verificando imports ---"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
from unsloth import FastLanguageModel
print(f'  Unsloth: OK')
"

# --- Check data exists ---
echo ""
echo "--- Verificando datos v2 ---"
for f in data/processed/cpt_train.jsonl data/processed/cpt_val.jsonl \
         data/processed/sft_v2_train.jsonl data/processed/sft_v2_val.jsonl; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        size=$(du -h "$f" | cut -f1)
        echo "  $f: $lines registros ($size)"
    else
        echo "  [ERROR] $f no encontrado! Ejecutar prepare_instructions_v2.py primero."
        exit 1
    fi
done

# --- Phase 1: CPT v2 (3 epochs) ---
echo ""
echo "============================================================"
echo "  FASE 1: Continual Pre-Training v2 (3 epochs)"
echo "  Config: configs/pretrain_v2_config.yaml"
echo "  Estimado: ~10h en RTX 4070 8GB"
echo "============================================================"
echo ""

START_CPT=$(date +%s)
python3 src/train_cpt.py --config configs/pretrain_v2_config.yaml
END_CPT=$(date +%s)
CPT_MINS=$(( (END_CPT - START_CPT) / 60 ))

echo ""
echo "  CPT v2 completado en ${CPT_MINS} minutos."
echo ""

# --- Verify CPT output ---
if [ ! -d "checkpoints/cpt_v2/final" ]; then
    echo "[ERROR] CPT v2 no produjo checkpoints/cpt_v2/final. Abortando."
    exit 1
fi

# --- Phase 2: SFT v2 (249K instructions, 2 epochs) ---
echo "============================================================"
echo "  FASE 2: SFT v2 (249K instructions, 2 epochs)"
echo "  Config: configs/sft_v2_config.yaml"
echo "  Estimado: ~7h en RTX 4070 8GB"
echo "============================================================"
echo ""

START_SFT=$(date +%s)
python3 src/train_sft.py --config configs/sft_v2_config.yaml
END_SFT=$(date +%s)
SFT_MINS=$(( (END_SFT - START_SFT) / 60 ))

echo ""
echo "  SFT v2 completado en ${SFT_MINS} minutos."
echo ""

# --- Summary ---
TOTAL_MINS=$(( CPT_MINS + SFT_MINS ))
echo "============================================================"
echo "  ENTRENAMIENTO v2 COMPLETO"
echo "============================================================"
echo "  CPT v2 (3ep): ${CPT_MINS} min"
echo "  SFT v2 (2ep): ${SFT_MINS} min"
echo "  Total: ${TOTAL_MINS} min"
echo ""
echo "  Checkpoints:"
echo "    CPT merged:  checkpoints/cpt_v2/final/"
echo "    SFT merged:  checkpoints/sft_v2/final/"
echo "    GGUF:        checkpoints/sft_v2/gguf/"
echo ""
echo "  Siguiente paso:"
echo "    1. Probar: python3 src/inference.py --model checkpoints/sft_v2/final"
echo "    2. Evaluar: python src/evaluate.py --config configs/eval_v2_config.yaml"
echo "    3. Comparar vs v1"
echo "============================================================"

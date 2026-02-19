#!/bin/bash
# =============================================================================
# GuaraniLM — Training Pipeline (WSL2)
#
# Ejecutar desde WSL2:
#   cd /mnt/c/Users/skyva/Documents/github-contributions/guarani-lm
#   bash run_training.sh
#
# Requisitos: NVIDIA GPU con drivers instalados, CUDA accesible desde WSL2
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  GuaraniLM — Training Pipeline"
echo "  Directorio: $PROJECT_DIR"
echo "============================================================"

# --- Check GPU ---
echo ""
echo "--- Verificando GPU ---"
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi no encontrado. Asegurate de que los drivers NVIDIA estan instalados en WSL2."
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

# Core training deps
pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -1
pip install datasets transformers accelerate peft bitsandbytes trl 2>&1 | tail -1
pip install pyyaml scikit-learn sacrebleu 2>&1 | tail -1

echo ""
echo "--- Verificando imports ---"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
from unsloth import FastLanguageModel
print(f'  Unsloth: OK')
from trl import SFTTrainer, SFTConfig
print(f'  TRL: OK')
"

# --- Check data exists ---
echo ""
echo "--- Verificando datos ---"
for f in data/processed/cpt_train.jsonl data/processed/cpt_val.jsonl \
         data/processed/sft_train.jsonl data/processed/sft_val.jsonl; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        size=$(du -h "$f" | cut -f1)
        echo "  $f: $lines registros ($size)"
    else
        echo "  [WARN] $f no encontrado!"
    fi
done

# --- Phase 1: CPT ---
echo ""
echo "============================================================"
echo "  FASE 1: Continual Pre-Training (CPT)"
echo "  Config: configs/pretrain_config.yaml"
echo "============================================================"
echo ""

START_CPT=$(date +%s)
python3 src/train_cpt.py --config configs/pretrain_config.yaml
END_CPT=$(date +%s)
CPT_MINS=$(( (END_CPT - START_CPT) / 60 ))

echo ""
echo "  CPT completado en ${CPT_MINS} minutos."
echo ""

# --- Verify CPT output ---
if [ ! -d "checkpoints/cpt/final" ]; then
    echo "[ERROR] CPT no produjo checkpoints/cpt/final. Abortando."
    exit 1
fi

# --- Phase 2: SFT ---
echo "============================================================"
echo "  FASE 2: Supervised Fine-Tuning (SFT)"
echo "  Config: configs/sft_config.yaml"
echo "============================================================"
echo ""

START_SFT=$(date +%s)
python3 src/train_sft.py --config configs/sft_config.yaml
END_SFT=$(date +%s)
SFT_MINS=$(( (END_SFT - START_SFT) / 60 ))

echo ""
echo "  SFT completado en ${SFT_MINS} minutos."
echo ""

# --- Summary ---
TOTAL_MINS=$(( CPT_MINS + SFT_MINS ))
echo "============================================================"
echo "  ENTRENAMIENTO COMPLETO"
echo "============================================================"
echo "  CPT: ${CPT_MINS} min"
echo "  SFT: ${SFT_MINS} min"
echo "  Total: ${TOTAL_MINS} min"
echo ""
echo "  Checkpoints:"
echo "    CPT merged:  checkpoints/cpt/final/"
echo "    SFT merged:  checkpoints/sft/final/"
echo "    GGUF:        checkpoints/sft/gguf/"
echo ""
echo "  Siguiente paso:"
echo "    1. Probar: python3 src/inference.py --model checkpoints/sft/final"
echo "    2. Evaluar: python3 src/evaluate.py --config configs/eval_config.yaml"
echo "    3. Publicar en HuggingFace"
echo "============================================================"

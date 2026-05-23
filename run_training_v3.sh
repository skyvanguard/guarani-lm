#!/bin/bash
# =============================================================================
# GuaraniLM v3 — Training Pipeline (Docker)
#
# Runs INSIDE the container created by docker-compose:
#   docker compose run --rm trainer bash run_training_v3.sh
#
# Changes vs v2:
#   - Base model: Qwen2.5-0.5B -> Qwen2.5-3B (6x params)
#   - CPT: 2 epochs (was 3) — bigger model converges faster
#   - LoRA: r=64 for CPT, r=32 for SFT (was 32/32)
#   - max_seq_length: 2048 (was 1024)
#   - Data mounted RO at /data (from D:/guarani-lm/data)
#   - Estimated time: ~18-22h total on RTX 5070 Ti 12GB
# =============================================================================

set -euo pipefail

PROJECT_DIR="/workspace"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  GuaraniLM v3 — Training Pipeline (Docker)"
echo "  Project dir: $PROJECT_DIR"
echo "============================================================"

# --- Check GPU ---
echo ""
echo "--- Verifying GPU ---"
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found — GPU passthrough is not working."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# --- Verify imports ---
echo "--- Verifying Python stack ---"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
from unsloth import FastLanguageModel
print(f'  Unsloth: OK')
import transformers, peft, trl, bitsandbytes
print(f'  transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__}')
"

# --- Check data ---
echo ""
echo "--- Verifying datasets at data/processed/ ---"
for f in data/processed/cpt_train.jsonl data/processed/cpt_val.jsonl \
         data/processed/sft_v2_train.jsonl data/processed/sft_v2_val.jsonl; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        size=$(du -h "$f" | cut -f1)
        echo "  $f: $lines records ($size)"
    else
        echo "  [ERROR] $f not found in workspace! Did you copy data from D:/guarani-lm/data/processed/ ?"
        exit 1
    fi
done

mkdir -p checkpoints logs

# --- Phase 1: CPT v3 (2 epochs) ---
echo ""
echo "============================================================"
echo "  PHASE 1: Continual Pre-Training v3 (Qwen2.5-3B, 2 epochs)"
echo "  Config: configs/pretrain_v3_config.yaml"
echo "  Estimated: ~12h on RTX 5070 Ti 12GB"
echo "============================================================"
echo ""

START_CPT=$(date +%s)
python3 src/train_cpt.py --config configs/pretrain_v3_config.yaml 2>&1 | tee logs/cpt_v3.log
END_CPT=$(date +%s)
CPT_MINS=$(( (END_CPT - START_CPT) / 60 ))

echo ""
echo "  CPT v3 completed in ${CPT_MINS} minutes."
echo ""

if [ ! -d "checkpoints/cpt_v3/final" ]; then
    echo "[ERROR] CPT v3 did not produce checkpoints/cpt_v3/final. Aborting."
    exit 1
fi

# --- Phase 2: SFT v3 (236K instructions, 2 epochs) ---
echo "============================================================"
echo "  PHASE 2: SFT v3 (236K instructions, 2 epochs)"
echo "  Config: configs/sft_v3_config.yaml"
echo "  Estimated: ~8h on RTX 5070 Ti 12GB"
echo "============================================================"
echo ""

START_SFT=$(date +%s)
python3 src/train_sft.py --config configs/sft_v3_config.yaml 2>&1 | tee logs/sft_v3.log
END_SFT=$(date +%s)
SFT_MINS=$(( (END_SFT - START_SFT) / 60 ))

echo ""
echo "  SFT v3 completed in ${SFT_MINS} minutes."
echo ""

# --- Summary ---
TOTAL_MINS=$(( CPT_MINS + SFT_MINS ))
echo "============================================================"
echo "  v3 TRAINING COMPLETE"
echo "============================================================"
echo "  CPT v3 (2ep): ${CPT_MINS} min"
echo "  SFT v3 (2ep): ${SFT_MINS} min"
echo "  Total: ${TOTAL_MINS} min"
echo ""
echo "  Checkpoints:"
echo "    CPT merged:  checkpoints/cpt_v3/final/"
echo "    SFT merged:  checkpoints/sft_v3/final/"
echo ""
echo "  Next steps:"
echo "    1. Eval: python3 src/evaluate.py --config configs/eval_v3_config.yaml"
echo "    2. Compare BLEU/chrF2 vs v2"
echo "    3. Export GGUF for Ollama"
echo "============================================================"

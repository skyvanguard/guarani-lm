# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

GuaraniLM is the first open-source generative (decoder-only) language model for Guaraní and Jopará, built as a QLoRA fine-tune on top of `unsloth/Qwen2.5-0.5B`. Training happens in two phases (CPT then SFT) and produces HuggingFace + GGUF (Ollama) checkpoints. Target hardware is a single ~8GB VRAM GPU (RTX 3070/4070), and the full training script is designed to run from WSL2 on Windows.

The repo carries **two training generations in parallel**: v1 (`pretrain_config.yaml`, `sft_config.yaml`, `checkpoints/cpt/`, `checkpoints/sft/`) and v2 (`*_v2_config.yaml`, `checkpoints/cpt_v2/`, `checkpoints/sft_v2/`, ~249K SFT instructions, 3 CPT epochs). When asked to "train" or "eval", confirm which generation the user means before touching configs.

## Common commands

Install (use the `train` extras for actual training — it pulls `unsloth`, `bitsandbytes`, `flash-attn`):
```bash
pip install -e .          # base
pip install -e ".[train]" # GPU training
pip install -e ".[dev]"   # pytest + ruff
```

Training (run each phase separately, or use the pipeline script):
```bash
# v1
python src/train_cpt.py --config configs/pretrain_config.yaml
python src/train_sft.py --config configs/sft_config.yaml

# v2 (full pipeline with GPU check, data check, CPT -> SFT)
bash run_training_v2.sh
```

Evaluation, inference, tests, lint:
```bash
python src/evaluate.py --config configs/eval_v2_config.yaml
python src/inference.py --model checkpoints/sft_v2/final
pytest                          # full suite
pytest tests/test_normalize.py  # single file
pytest tests/test_normalize.py::test_name  # single test
ruff check .
```

Data pipeline scripts live in `scripts/` and are invoked individually (there is no orchestrator): `download_data.py`, `clean_wikipedia.py`, `clean_culturax.py`, `normalize_guarani.py`, `prepare_parallel.py`, `augment_nllb.py`, `merge_datasets.py`, `prepare_instructions.py` / `prepare_instructions_v2.py`, `prepare_test_sets.py`, `count_tokens.py`, `tokenizer_analysis.py`.

## Architecture notes

- **`src/` is flat, not a package tree.** Each entrypoint (`train_cpt.py`, `train_sft.py`, `evaluate.py`, `inference.py`) is a CLI that takes `--config <yaml>` and is the *only* thing that reads that YAML. Shared logic lives in `guarani_utils.py` and `prompt_templates.py` — put cross-phase code there, not in the training scripts.
- **Configs are the contract.** `configs/*.yaml` drive model name, LoRA rank/targets, batch sizes, data paths (`data/processed/*.jsonl`), and output dirs. Expected data layout: `data/processed/cpt_train.jsonl`, `cpt_val.jsonl`, `sft_v2_train.jsonl`, `sft_v2_val.jsonl` (see `run_training_v2.sh` for the full list). The `data/` directory is gitignored and must exist before training.
- **Two-phase training flow.** Phase 1 (CPT) does QLoRA 4-bit on Qwen2.5-0.5B with embeddings + all linear layers trained, producing `checkpoints/cpt[_v2]/final/` (merged). Phase 2 (SFT) loads *that merged checkpoint* and applies a second LoRA adapter over ChatML-formatted instructions, producing `checkpoints/sft[_v2]/final/` plus a GGUF export under `checkpoints/sft[_v2]/gguf/`. Breaking the CPT output contract breaks SFT — keep `output_dir` and the `final/` subdir convention intact.
- **v1 vs v2 config differences** are the knobs most likely to change: CPT epochs (1 → 3), SFT dataset size and epochs, LoRA rank. VRAM is tight at 8GB, so `per_device_train_batch_size`, `gradient_accumulation_steps`, `max_seq_length`, and `gradient_checkpointing` in the YAMLs are already tuned — adjust deliberately.
- **Normalization matters for Guaraní.** `scripts/normalize_guarani.py` and `src/guarani_utils.py` handle Guaraní-specific orthography (puso ', nasal tildes). Tests in `tests/test_normalize.py` pin that behavior — run them after any change to normalization or tokenizer handling.
- **Evaluation outputs** are written as timestamped JSON into `eval/results/`. The model card in `docs/model_card.md` is the canonical place to record scores.

## Environment

Developer machine is Windows 11; training runs from WSL2 (`run_training_v2.sh` assumes `/mnt/c/...` paths and `nvidia-smi`). When editing shell scripts or paths, keep them POSIX — do not port to PowerShell.

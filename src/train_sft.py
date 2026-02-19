"""Supervised Fine-Tuning (SFT) script for GuaraniLM.

Fine-tunes the CPT checkpoint on instruction-following data in ChatML format.
Uses LoRA r=64 (no embed_tokens/lm_head) and NEFTune noise.

Usage (from project root, inside WSL2)::

    python src/train_sft.py --config configs/sft_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_sft")

# Add src/ to path for prompt_templates import
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    for section in ("model", "lora", "training", "data"):
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")
    return config


def format_conversations_chatml(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Convert a conversations list to a single ChatML text string."""
    conversations = example.get("conversations", [])

    try:
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        from prompt_templates import format_chatml
        # format_chatml adds a trailing generation prompt, remove it
        text = format_chatml(conversations)
        if text.endswith("<|im_start|>assistant\n"):
            text = text[: -len("<|im_start|>assistant\n")]

    return {"text": text}


def train(config: dict[str, Any]) -> None:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    sft_cfg = config.get("sft", {})

    # --- Load model (CPT checkpoint) ---
    logger.info("Loading CPT checkpoint: %s", model_cfg["name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Apply LoRA ---
    logger.info("Applying LoRA: r=%d, targets=%s", lora_cfg["r"], lora_cfg["target_modules"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        use_rslora=lora_cfg.get("use_rslora", False),
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # --- Load and format data ---
    train_path = Path(data_cfg["train_file"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info("Loading SFT data: %s", train_path)
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")

    eval_dataset = None
    val_path = data_cfg.get("validation_file")
    if val_path and Path(val_path).exists():
        logger.info("Loading validation data: %s", val_path)
        eval_dataset = load_dataset("json", data_files=str(val_path), split="train")

    # Format conversations -> ChatML text
    logger.info("Formatting conversations to ChatML...")

    def _fmt(example):
        return format_conversations_chatml(example, tokenizer)

    train_dataset = train_dataset.map(_fmt, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_fmt, remove_columns=eval_dataset.column_names)

    logger.info("Train: %d samples%s", len(train_dataset),
                f", Val: {len(eval_dataset)} samples" if eval_dataset else "")

    # --- Training config ---
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        report_to=train_cfg.get("report_to", "none"),
        optim="adamw_8bit",
        # SFT-specific
        packing=sft_cfg.get("packing", False),
        max_seq_length=data_cfg.get("max_length", 2048),
        dataset_text_field="text",
        neftune_noise_alpha=sft_cfg.get("neftune_noise_alpha"),
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    logger.info("Starting Supervised Fine-Tuning...")
    if sft_cfg.get("neftune_noise_alpha"):
        logger.info("NEFTune noise alpha: %s", sft_cfg["neftune_noise_alpha"])

    result = trainer.train()
    metrics = result.metrics
    logger.info("Training complete. Loss: %.4f", metrics.get("train_loss", 0))

    # --- Save ---
    logger.info("Saving adapter checkpoint...")
    trainer.save_model(str(output_dir / "final_adapter"))

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging adapter -> %s", final_dir)
    model.save_pretrained_merged(str(final_dir), tokenizer, save_method="merged_16bit")

    # Export GGUF for Ollama
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting GGUF (Q4_K_M) -> %s", gguf_dir)
    try:
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")
        logger.info("GGUF export complete!")
    except Exception as e:
        logger.warning("GGUF export failed (non-critical): %s", e)

    config_save = final_dir / "training_config.yaml"
    with open(config_save, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("SFT complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuaraniLM - SFT")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))

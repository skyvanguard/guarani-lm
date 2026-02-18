"""Supervised Fine-Tuning (SFT) script for GuaraniLM.

Fine-tunes the CPT checkpoint on instruction-following data in ChatML format.
Uses LoRA r=64 (no embed_tokens/lm_head) and NEFTune noise for better
generalization.

Usage::

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

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None  # type: ignore[assignment,misc]
    logger.warning(
        "Unsloth not installed. Install with: pip install 'guarani-lm[train]'"
    )

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

import torch
from datasets import load_dataset
from transformers import TrainingArguments

try:
    from trl import SFTTrainer
except ImportError:
    raise ImportError("trl is required. Install with: pip install trl")

from prompt_templates import format_chatml


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    for section in ("model", "lora", "training", "data"):
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")

    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Load the CPT checkpoint model and tokenizer using Unsloth.

    Parameters
    ----------
    config : dict[str, Any]
        Full configuration dictionary.

    Returns
    -------
    tuple
        ``(model, tokenizer)``
    """
    if FastLanguageModel is None:
        raise ImportError(
            "Unsloth is required for training. Install with: pip install 'guarani-lm[train]'"
        )

    model_cfg = config["model"]
    logger.info(
        "Loading CPT checkpoint: %s (4-bit=%s)",
        model_cfg["name"],
        model_cfg["load_in_4bit"],
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg.get("dtype"),
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def apply_lora(model: Any, config: dict[str, Any]) -> Any:
    """Apply LoRA adapters (no embed_tokens/lm_head for SFT).

    Parameters
    ----------
    model : Any
        Model loaded from the CPT checkpoint.
    config : dict[str, Any]
        Full config with ``lora`` section.

    Returns
    -------
    Any
        Model with LoRA adapters.
    """
    if FastLanguageModel is None:
        raise ImportError("Unsloth is required for LoRA configuration.")

    lora_cfg = config["lora"]

    logger.info(
        "Applying LoRA: r=%d, alpha=%d, dropout=%.3f, targets=%s",
        lora_cfg["r"],
        lora_cfg["lora_alpha"],
        lora_cfg["lora_dropout"],
        lora_cfg["target_modules"],
    )

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

    return model


# ---------------------------------------------------------------------------
# ChatML data formatting
# ---------------------------------------------------------------------------

def format_dataset_chatml(
    example: dict[str, Any],
    tokenizer: Any,
) -> dict[str, str]:
    """Format a single example into ChatML using the tokenizer's template.

    Expects examples with a ``messages`` field containing a list of
    ``{"role": ..., "content": ...}`` dicts.

    Parameters
    ----------
    example : dict[str, Any]
        A dataset row with ``messages`` key.
    tokenizer : Any
        Tokenizer with ``apply_chat_template`` support.

    Returns
    -------
    dict[str, str]
        Dict with ``"text"`` key containing the formatted ChatML string.
    """
    messages = example.get("messages", [])

    # Try tokenizer's built-in chat template first, fall back to manual
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = format_chatml(messages)
        # Remove the trailing generation prompt since we have the full conversation
        if text.endswith("<|im_start|>assistant\n"):
            text = text[: -len("<|im_start|>assistant\n")]

    return {"text": text}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[Any, Any]:
    """Load and format SFT training data from JSONL.

    Parameters
    ----------
    config : dict[str, Any]
        Full config with ``data`` section.
    tokenizer : Any
        Tokenizer for chat template formatting.

    Returns
    -------
    tuple
        ``(train_dataset, eval_dataset)`` — eval may be ``None``.
    """
    data_cfg = config["data"]

    train_path = Path(data_cfg["train_file"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info("Loading SFT training data from: %s", train_path)
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")

    eval_dataset = None
    val_path = data_cfg.get("validation_file")
    if val_path and Path(val_path).exists():
        logger.info("Loading validation data from: %s", val_path)
        eval_dataset = load_dataset("json", data_files=str(val_path), split="train")

    # Format into ChatML
    logger.info("Formatting dataset with ChatML template...")

    def _fmt(example: dict[str, Any]) -> dict[str, str]:
        return format_dataset_chatml(example, tokenizer)

    train_dataset = train_dataset.map(_fmt, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_fmt, remove_columns=eval_dataset.column_names)

    logger.info(
        "Loaded %d training examples%s",
        len(train_dataset),
        f", {len(eval_dataset)} validation examples" if eval_dataset else "",
    )

    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict[str, Any]) -> None:
    """Run the full SFT training pipeline.

    Parameters
    ----------
    config : dict[str, Any]
        Full configuration dictionary.
    """
    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    if wandb is not None and config["training"].get("report_to") == "wandb":
        wandb.init(
            project=wandb_cfg.get("project", "guarani-lm"),
            name=wandb_cfg.get("run_name", "sft-run"),
            config=config,
        )

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA
    model = apply_lora(model, config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    # Load data
    train_dataset, eval_dataset = load_training_data(config, tokenizer)

    # Build training arguments
    train_cfg = config["training"]
    data_cfg = config["data"]
    sft_cfg = config.get("sft", {})

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
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
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        report_to=train_cfg.get("report_to", "none"),
        optim="adamw_8bit",
        neftune_noise_alpha=sft_cfg.get("neftune_noise_alpha"),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        packing=sft_cfg.get("packing", False),
        max_seq_length=data_cfg.get("max_length", 2048),
        dataset_text_field="text",
    )

    # Train
    logger.info("Starting Supervised Fine-Tuning...")
    if sft_cfg.get("neftune_noise_alpha"):
        logger.info("NEFTune noise alpha: %s", sft_cfg["neftune_noise_alpha"])

    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", json.dumps(metrics, indent=2))

    # Save adapter checkpoint
    logger.info("Saving final adapter checkpoint...")
    trainer.save_model(str(output_dir / "final_adapter"))

    # Save merged model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging adapter weights and saving to: %s", final_dir)
    model.save_pretrained_merged(
        str(final_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    # Also export GGUF for Ollama (Q4_K_M quantization)
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting GGUF (Q4_K_M) to: %s", gguf_dir)
    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        logger.info("GGUF export complete!")
    except Exception as e:
        logger.warning("GGUF export failed (non-critical): %s", e)

    # Save training config
    config_save_path = final_dir / "training_config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("SFT training pipeline complete!")

    if wandb is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GuaraniLM — Supervised Fine-Tuning (SFT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to the YAML config file (default: configs/sft_config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)

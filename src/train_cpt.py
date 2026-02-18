"""Continual Pre-Training (CPT) script for GuaraniLM.

Teaches Guarani to a base model (Qwen2.5-0.5B) using QLoRA 4-bit with
Unsloth acceleration.  Trains embeddings + lm_head with a lower learning
rate and all linear layers with a higher rate.

Usage::

    python src/train_cpt.py --config configs/pretrain_config.yaml
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
logger = logging.getLogger("train_cpt")

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
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
except ImportError:
    from trl import SFTTrainer  # type: ignore[assignment]


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

    # Validate required sections
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
    """Load the base model and tokenizer using Unsloth FastLanguageModel.

    Parameters
    ----------
    config : dict[str, Any]
        Full configuration dictionary.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` ready for LoRA patching.
    """
    if FastLanguageModel is None:
        raise ImportError(
            "Unsloth is required for training. Install with: pip install 'guarani-lm[train]'"
        )

    model_cfg = config["model"]
    logger.info("Loading model: %s (4-bit=%s)", model_cfg["name"], model_cfg["load_in_4bit"])

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
    """Apply LoRA adapters to the model.

    Parameters
    ----------
    model : Any
        Base model loaded via Unsloth.
    config : dict[str, Any]
        Full config with ``lora`` section.

    Returns
    -------
    Any
        Model with LoRA adapters attached.
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
# Dual learning rate optimizer
# ---------------------------------------------------------------------------

def build_optimizer_groups(
    model: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build parameter groups with dual learning rates.

    Embedding and lm_head parameters get a lower learning rate to
    avoid catastrophic forgetting of existing token representations.

    Parameters
    ----------
    model : Any
        Model with LoRA adapters.
    config : dict[str, Any]
        Full config with ``training`` section.

    Returns
    -------
    list[dict[str, Any]]
        Parameter groups for the optimizer.
    """
    train_cfg = config["training"]
    general_lr = train_cfg["learning_rate"]
    embed_lr = train_cfg.get("embedding_learning_rate", general_lr / 10)

    embedding_params: list[torch.nn.Parameter] = []
    general_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed_tokens" in name or "lm_head" in name:
            embedding_params.append(param)
        else:
            general_params.append(param)

    logger.info(
        "Optimizer groups: %d general params (lr=%.2e), %d embedding params (lr=%.2e)",
        len(general_params),
        general_lr,
        len(embedding_params),
        embed_lr,
    )

    groups: list[dict[str, Any]] = [
        {"params": general_params, "lr": general_lr},
        {"params": embedding_params, "lr": embed_lr},
    ]

    return groups


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(config: dict[str, Any]) -> Any:
    """Load training and optional validation datasets from JSONL.

    Parameters
    ----------
    config : dict[str, Any]
        Full config with ``data`` section.

    Returns
    -------
    tuple
        ``(train_dataset, eval_dataset)`` — eval may be ``None``.
    """
    data_cfg = config["data"]

    train_path = Path(data_cfg["train_file"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info("Loading training data from: %s", train_path)
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")

    eval_dataset = None
    val_path = data_cfg.get("validation_file")
    if val_path and Path(val_path).exists():
        logger.info("Loading validation data from: %s", val_path)
        eval_dataset = load_dataset("json", data_files=str(val_path), split="train")

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
    """Run the full CPT training pipeline.

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
            name=wandb_cfg.get("run_name", "cpt-run"),
            config=config,
        )

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

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
    train_dataset, eval_dataset = load_training_data(config)

    # Build training arguments
    train_cfg = config["training"]
    data_cfg = config["data"]

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
    )

    # Build optimizer with dual learning rates
    optimizer_groups = build_optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=train_cfg["weight_decay"],
    )

    # Create trainer
    text_field = data_cfg.get("text_field", "text")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        packing=data_cfg.get("packing", True),
        max_seq_length=data_cfg.get("max_length", 2048),
        dataset_text_field=text_field,
        optimizers=(optimizer, None),  # custom optimizer, default scheduler
    )

    # Train
    logger.info("Starting Continual Pre-Training...")
    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", json.dumps(metrics, indent=2))

    # Save checkpoint
    logger.info("Saving final checkpoint...")
    trainer.save_model(str(output_dir / "final_adapter"))

    # Save merged model (adapter + base)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merging adapter weights and saving to: %s", final_dir)
    model.save_pretrained_merged(
        str(final_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    # Save training config alongside the model
    config_save_path = final_dir / "training_config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("CPT training pipeline complete!")

    if wandb is not None:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GuaraniLM — Continual Pre-Training (CPT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_config.yaml",
        help="Path to the YAML config file (default: configs/pretrain_config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)

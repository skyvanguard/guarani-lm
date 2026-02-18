"""Evaluation script for GuaraniLM.

Evaluates the model on translation, sentiment, classification, and perplexity
tasks.  Optionally compares against baselines (NLLB-200, gn-bert, Qwen base).

Usage::

    python src/evaluate.py --config configs/eval_config.yaml
    python src/evaluate.py --config configs/eval_config.yaml --tasks translation_gn_es sentiment
    python src/evaluate.py --config configs/eval_config.yaml --baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# Core dependencies
# ---------------------------------------------------------------------------

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Optional metric dependencies
# ---------------------------------------------------------------------------

try:
    import sacrebleu
except ImportError:
    sacrebleu = None  # type: ignore[assignment]
    logger.warning("sacrebleu not installed — BLEU/chrF metrics unavailable")

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    accuracy_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    logger.warning("scikit-learn not installed — classification metrics unavailable")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load evaluation config from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_eval(
    model_name: str,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048,
) -> tuple[Any, Any]:
    """Load model and tokenizer for evaluation.

    Tries Unsloth first (for 4-bit), falls back to standard transformers.

    Parameters
    ----------
    model_name : str
        Model path or HuggingFace hub ID.
    load_in_4bit : bool
        Whether to use 4-bit quantization.
    max_seq_length : int
        Maximum sequence length.

    Returns
    -------
    tuple
        ``(model, tokenizer)``
    """
    try:
        from unsloth import FastLanguageModel

        logger.info("Loading model with Unsloth: %s", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        logger.info("Unsloth not available. Loading with transformers: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    do_sample: bool = False,
    batch_size: int = 8,
) -> list[str]:
    """Generate model responses for a list of prompts.

    Parameters
    ----------
    model : Any
        Language model.
    tokenizer : Any
        Associated tokenizer.
    prompts : list[str]
        Input prompts.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    do_sample : bool
        Whether to sample (False = greedy decoding).
    batch_size : int
        Number of prompts to process at once.

    Returns
    -------
    list[str]
        Generated response texts (prompt removed).
    """
    responses: list[str] = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            outputs = model.generate(**inputs, **gen_kwargs)

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            responses.append(text)

    return responses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU score using sacrebleu.

    Parameters
    ----------
    predictions : list[str]
        Model outputs.
    references : list[str]
        Reference translations.

    Returns
    -------
    float
        BLEU score (0-100).
    """
    if sacrebleu is None:
        logger.warning("sacrebleu not available — returning 0.0 for BLEU")
        return 0.0

    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score


def compute_chrf2(predictions: list[str], references: list[str]) -> float:
    """Compute corpus chrF2 score using sacrebleu.

    Parameters
    ----------
    predictions : list[str]
        Model outputs.
    references : list[str]
        Reference translations.

    Returns
    -------
    float
        chrF2 score (0-100).
    """
    if sacrebleu is None:
        logger.warning("sacrebleu not available — returning 0.0 for chrF2")
        return 0.0

    result = sacrebleu.corpus_chrf(predictions, [references], beta=2)
    return result.score


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute accuracy for classification tasks.

    Normalizes both predictions and references to lowercase before
    comparing.

    Parameters
    ----------
    predictions : list[str]
        Model predicted labels.
    references : list[str]
        Gold labels.

    Returns
    -------
    float
        Accuracy (0-1).
    """
    if accuracy_score is None:
        logger.warning("sklearn not available — returning 0.0 for accuracy")
        return 0.0

    preds_norm = [p.strip().lower() for p in predictions]
    refs_norm = [r.strip().lower() for r in references]
    return accuracy_score(refs_norm, preds_norm)


def compute_macro_f1(predictions: list[str], references: list[str]) -> float:
    """Compute macro-averaged F1 for classification tasks.

    Parameters
    ----------
    predictions : list[str]
        Model predicted labels.
    references : list[str]
        Gold labels.

    Returns
    -------
    float
        Macro F1 (0-1).
    """
    if f1_score is None:
        logger.warning("sklearn not available — returning 0.0 for macro F1")
        return 0.0

    preds_norm = [p.strip().lower() for p in predictions]
    refs_norm = [r.strip().lower() for r in references]
    return f1_score(refs_norm, preds_norm, average="macro", zero_division=0)


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int = 2048,
    batch_size: int = 8,
) -> float:
    """Compute perplexity of the model on a list of texts.

    Parameters
    ----------
    model : Any
        Language model.
    tokenizer : Any
        Tokenizer.
    texts : list[str]
        Texts to evaluate.
    max_length : int
        Maximum sequence length.
    batch_size : int
        Batch size.

    Returns
    -------
    float
        Perplexity (lower is better).
    """
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # Compute per-token loss excluding padding
            loss = outputs.loss
            # Count non-padding tokens
            attention_mask = inputs["attention_mask"]
            n_tokens = attention_mask.sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

METRIC_FUNCTIONS: dict[str, Any] = {
    "bleu": compute_bleu,
    "chrf2": compute_chrf2,
    "accuracy": compute_accuracy,
    "macro_f1": compute_macro_f1,
}


def evaluate_generation_task(
    model: Any,
    tokenizer: Any,
    task_name: str,
    task_config: dict[str, Any],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a generation-based task (translation, sentiment, classification).

    Parameters
    ----------
    model : Any
        Model to evaluate.
    tokenizer : Any
        Tokenizer.
    task_name : str
        Name of the task.
    task_config : dict[str, Any]
        Task-specific config (test_file, metrics, etc.).
    eval_config : dict[str, Any]
        Global evaluation config (batch_size, max_new_tokens, etc.).

    Returns
    -------
    dict[str, Any]
        Results dict with metric scores and metadata.
    """
    test_path = Path(task_config["test_file"])
    if not test_path.exists():
        logger.warning("Test file not found: %s — skipping %s", test_path, task_name)
        return {"error": f"Test file not found: {test_path}"}

    logger.info("Evaluating task: %s", task_name)
    dataset = load_dataset("json", data_files=str(test_path), split="train")
    logger.info("  Loaded %d test examples", len(dataset))

    # Build prompts — expects "prompt" and "reference" fields
    prompts = dataset["prompt"]
    references = dataset["reference"]

    # Generate responses
    start_time = time.time()
    predictions = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens=eval_config.get("max_new_tokens", 256),
        temperature=eval_config.get("temperature", 0.1),
        top_p=eval_config.get("top_p", 0.9),
        do_sample=eval_config.get("do_sample", False),
        batch_size=eval_config.get("batch_size", 8),
    )
    elapsed = time.time() - start_time

    # Compute metrics
    results: dict[str, Any] = {
        "task": task_name,
        "num_examples": len(dataset),
        "generation_time_s": round(elapsed, 2),
    }

    metrics = task_config.get("metrics", [])
    for metric_name in metrics:
        fn = METRIC_FUNCTIONS.get(metric_name)
        if fn is not None:
            score = fn(predictions, references)
            results[metric_name] = round(score, 4)
            logger.info("  %s: %.4f", metric_name, score)

    # Save example predictions
    results["examples"] = [
        {"prompt": p, "prediction": pred, "reference": ref}
        for p, pred, ref in zip(prompts[:5], predictions[:5], references[:5])
    ]

    return results


def evaluate_perplexity_task(
    model: Any,
    tokenizer: Any,
    task_name: str,
    task_config: dict[str, Any],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate perplexity on Guarani text.

    Parameters
    ----------
    model : Any
        Model to evaluate.
    tokenizer : Any
        Tokenizer.
    task_name : str
        Name of the task.
    task_config : dict[str, Any]
        Task config with ``test_file``.
    eval_config : dict[str, Any]
        Global evaluation config.

    Returns
    -------
    dict[str, Any]
        Results with perplexity score.
    """
    test_path = Path(task_config["test_file"])
    if not test_path.exists():
        logger.warning("Test file not found: %s — skipping %s", test_path, task_name)
        return {"error": f"Test file not found: {test_path}"}

    logger.info("Evaluating task: %s (perplexity)", task_name)
    dataset = load_dataset("json", data_files=str(test_path), split="train")
    logger.info("  Loaded %d test texts", len(dataset))

    texts = dataset["text"]

    start_time = time.time()
    ppl = compute_perplexity(
        model,
        tokenizer,
        texts,
        batch_size=eval_config.get("batch_size", 8),
    )
    elapsed = time.time() - start_time

    logger.info("  perplexity: %.2f", ppl)

    return {
        "task": task_name,
        "num_examples": len(dataset),
        "perplexity": round(ppl, 2),
        "eval_time_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline_nllb(
    baseline_config: dict[str, Any],
    tasks_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate NLLB-200 baseline on translation tasks.

    Parameters
    ----------
    baseline_config : dict[str, Any]
        Baseline model config.
    tasks_config : dict[str, Any]
        All task configurations.

    Returns
    -------
    dict[str, Any]
        Baseline results per task.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as AT
    except ImportError:
        return {"error": "transformers not available"}

    model_name = baseline_config["model"]
    logger.info("Loading NLLB baseline: %s", model_name)

    tokenizer = AT.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results: dict[str, Any] = {"model": model_name}

    lang_map = {"grn": "grn_Latn", "spa": "spa_Latn"}

    for task_name in baseline_config.get("tasks", []):
        task_cfg = tasks_config.get(task_name, {})
        if not task_cfg.get("enabled", True):
            continue

        test_path = Path(task_cfg["test_file"])
        if not test_path.exists():
            results[task_name] = {"error": f"File not found: {test_path}"}
            continue

        dataset = load_dataset("json", data_files=str(test_path), split="train")

        src_lang = lang_map.get(task_cfg.get("source_lang", ""), task_cfg.get("source_lang", ""))
        tgt_lang = lang_map.get(task_cfg.get("target_lang", ""), task_cfg.get("target_lang", ""))

        tokenizer.src_lang = src_lang
        sources = dataset["source"] if "source" in dataset.column_names else dataset["prompt"]
        references = dataset["reference"]

        predictions: list[str] = []
        device = next(model.parameters()).device

        for text in tqdm(sources, desc=f"NLLB {task_name}"):
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_new_tokens=256,
                )
            pred = tokenizer.decode(generated[0], skip_special_tokens=True)
            predictions.append(pred)

        task_results: dict[str, Any] = {"num_examples": len(dataset)}
        for metric_name in task_cfg.get("metrics", []):
            fn = METRIC_FUNCTIONS.get(metric_name)
            if fn:
                score = fn(predictions, references)
                task_results[metric_name] = round(score, 4)

        results[task_name] = task_results

    return results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def print_results_table(all_results: dict[str, Any]) -> None:
    """Print evaluation results as a formatted table to stdout.

    Parameters
    ----------
    all_results : dict[str, Any]
        Results dictionary with task names as keys.
    """
    print("\n" + "=" * 70)
    print("  GuaraniLM Evaluation Results")
    print("=" * 70)

    # Main model results
    main_results = all_results.get("guarani_lm", {})
    if main_results:
        print(f"\n  Model: {all_results.get('model_name', 'unknown')}")
        print(f"  Date:  {all_results.get('timestamp', 'unknown')}")
        print("-" * 70)
        print(f"  {'Task':<25} {'Metric':<15} {'Score':<15}")
        print("-" * 70)

        for task_name, task_result in main_results.items():
            if isinstance(task_result, dict) and "error" not in task_result:
                for key, value in task_result.items():
                    if key in ("task", "num_examples", "generation_time_s", "eval_time_s", "examples"):
                        continue
                    print(f"  {task_name:<25} {key:<15} {value:<15}")

    # Baseline results
    baselines = all_results.get("baselines", {})
    if baselines:
        print("\n" + "-" * 70)
        print("  Baselines")
        print("-" * 70)
        for baseline_name, baseline_result in baselines.items():
            model_id = baseline_result.get("model", baseline_name)
            print(f"\n  [{model_id}]")
            for task_name, task_result in baseline_result.items():
                if task_name == "model" or not isinstance(task_result, dict):
                    continue
                for key, value in task_result.items():
                    if key in ("num_examples", "error"):
                        continue
                    print(f"    {task_name:<23} {key:<15} {value:<15}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(config: dict[str, Any], task_filter: list[str] | None = None, run_baselines: bool = False) -> None:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    config : dict[str, Any]
        Evaluation configuration.
    task_filter : list[str] | None
        Optional list of task names to evaluate. If ``None``, all enabled tasks run.
    run_baselines : bool
        Whether to also evaluate baseline models.
    """
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    tasks_cfg = config.get("tasks", {})

    # Load model
    model, tokenizer = load_model_for_eval(
        model_name=model_cfg["name"],
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        max_seq_length=model_cfg.get("max_seq_length", 2048),
    )

    # Evaluate each task
    all_results: dict[str, Any] = {
        "model_name": model_cfg["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "guarani_lm": {},
    }

    for task_name, task_cfg in tasks_cfg.items():
        if not task_cfg.get("enabled", True):
            continue
        if task_filter and task_name not in task_filter:
            continue

        if "perplexity" in task_cfg.get("metrics", []):
            result = evaluate_perplexity_task(model, tokenizer, task_name, task_cfg, eval_cfg)
        else:
            result = evaluate_generation_task(model, tokenizer, task_name, task_cfg, eval_cfg)

        all_results["guarani_lm"][task_name] = result

    # Baselines
    if run_baselines:
        baselines_cfg = config.get("baselines", {})
        all_results["baselines"] = {}

        for baseline_name, baseline_cfg in baselines_cfg.items():
            logger.info("Evaluating baseline: %s", baseline_name)
            if baseline_name == "nllb_200":
                result = evaluate_baseline_nllb(baseline_cfg, tasks_cfg)
                all_results["baselines"][baseline_name] = result
            elif baseline_name == "qwen_base":
                # Evaluate Qwen base perplexity
                qwen_model, qwen_tok = load_model_for_eval(
                    baseline_cfg["model"],
                    load_in_4bit=True,
                )
                baseline_results: dict[str, Any] = {"model": baseline_cfg["model"]}
                for task_name in baseline_cfg.get("tasks", []):
                    task_cfg = tasks_cfg.get(task_name, {})
                    if task_cfg and "perplexity" in task_cfg.get("metrics", []):
                        result = evaluate_perplexity_task(
                            qwen_model, qwen_tok, task_name, task_cfg, eval_cfg,
                        )
                        baseline_results[task_name] = result
                all_results["baselines"][baseline_name] = baseline_results
                del qwen_model, qwen_tok
                torch.cuda.empty_cache()
            else:
                logger.warning("Unknown baseline type: %s — skipping", baseline_name)

    # Save results
    output_dir = Path(eval_cfg.get("output_dir", "eval/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_results_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)

    # Print table
    print_results_table(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GuaraniLM — Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to the YAML eval config (default: configs/eval_config.yaml)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional list of task names to evaluate (default: all enabled)",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        default=False,
        help="Also evaluate baseline models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_evaluation(cfg, task_filter=args.tasks, run_baselines=args.baselines)

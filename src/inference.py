"""Interactive inference demo for GuaraniLM.

Provides a REPL interface for translation, chat, and classification using
the fine-tuned model.

Usage::

    python src/inference.py --model checkpoints/sft/final --mode translate_gn_es
    python src/inference.py --model skyvanguard/guarani-lm-0.5b --mode chat
    python src/inference.py --model checkpoints/sft/final --mode classify --temperature 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from prompt_templates import build_messages, format_chatml, list_tasks

# ---------------------------------------------------------------------------
# Supported modes
# ---------------------------------------------------------------------------

MODES = {
    "translate_gn_es": {
        "task": "translate_gn_es",
        "description": "Traduccion Guarani -> Español",
        "prompt_hint": "Escribi texto en Guarani para traducir al Español",
    },
    "translate_es_gn": {
        "task": "translate_es_gn",
        "description": "Traduccion Español -> Guarani",
        "prompt_hint": "Escribi texto en Español para traducir al Guarani",
    },
    "chat": {
        "task": "chat",
        "description": "Chat bilingüe Guarani/Español/Jopara",
        "prompt_hint": "Conversa en Guarani, Español o Jopara",
    },
    "classify": {
        "task": "sentiment",
        "description": "Clasificacion de sentimiento",
        "prompt_hint": "Escribi un texto para analizar su sentimiento",
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    max_seq_length: int = 2048,
) -> tuple[Any, Any]:
    """Load model and tokenizer for inference.

    Tries Unsloth first for optimized inference, falls back to transformers.

    Parameters
    ----------
    model_path : str
        Local path or HuggingFace hub ID.
    max_seq_length : int
        Maximum sequence length.

    Returns
    -------
    tuple
        ``(model, tokenizer)``
    """
    try:
        from unsloth import FastLanguageModel

        logger.info("Loading model with Unsloth: %s", model_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        logger.info("Loading model with transformers: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    stream: bool = True,
) -> str:
    """Generate a response from a list of ChatML messages.

    Parameters
    ----------
    model : Any
        Language model.
    tokenizer : Any
        Tokenizer.
    messages : list[dict[str, str]]
        Conversation messages.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling parameter.
    top_k : int
        Top-k sampling parameter.
    repetition_penalty : float
        Repetition penalty.
    do_sample : bool
        Whether to sample.
    stream : bool
        Whether to stream output token by token.

    Returns
    -------
    str
        Generated response text.
    """
    # Format with ChatML
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = format_chatml(messages)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k

    streamer = None
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return response


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def print_header(mode_info: dict[str, str], gen_params: dict[str, Any]) -> None:
    """Print the REPL header with mode and generation info.

    Parameters
    ----------
    mode_info : dict[str, str]
        Mode description.
    gen_params : dict[str, Any]
        Generation parameters to display.
    """
    print("\n" + "=" * 60)
    print("  GuaraniLM — Interactive Inference")
    print("=" * 60)
    print(f"  Modo: {mode_info['description']}")
    print(f"  {mode_info['prompt_hint']}")
    print("-" * 60)
    print("  Parametros de generacion:")
    for key, value in gen_params.items():
        print(f"    {key}: {value}")
    print("-" * 60)
    print("  Comandos:")
    print("    /mode <name>  — Cambiar modo (translate_gn_es, translate_es_gn, chat, classify)")
    print("    /params       — Mostrar parametros actuales")
    print("    /temp <val>   — Cambiar temperatura")
    print("    /tokens <val> — Cambiar max tokens")
    print("    /quit         — Salir")
    print("=" * 60 + "\n")


def run_repl(
    model: Any,
    tokenizer: Any,
    *,
    mode: str = "chat",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> None:
    """Run the interactive REPL loop.

    Parameters
    ----------
    model : Any
        Loaded model.
    tokenizer : Any
        Tokenizer.
    mode : str
        Initial mode.
    max_new_tokens : int
        Default max tokens.
    temperature : float
        Default temperature.
    top_p : float
        Default top_p.
    top_k : int
        Default top_k.
    repetition_penalty : float
        Default repetition penalty.
    """
    current_mode = mode
    mode_info = MODES.get(current_mode, MODES["chat"])

    gen_params: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
    }

    # Chat history for multi-turn conversations
    chat_history: list[dict[str, str]] = []

    print_header(mode_info, gen_params)

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAguyje! Bye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit" or cmd == "/exit":
                print("Aguyje! Bye!")
                break

            elif cmd == "/mode":
                if len(parts) < 2 or parts[1] not in MODES:
                    print(f"  Modos disponibles: {', '.join(MODES.keys())}")
                else:
                    current_mode = parts[1]
                    mode_info = MODES[current_mode]
                    chat_history.clear()
                    print(f"  Modo cambiado a: {mode_info['description']}")

            elif cmd == "/params":
                print("  Parametros actuales:")
                for k, v in gen_params.items():
                    print(f"    {k}: {v}")

            elif cmd == "/temp":
                if len(parts) < 2:
                    print(f"  Temperatura actual: {gen_params['temperature']}")
                else:
                    try:
                        new_temp = float(parts[1])
                        gen_params["temperature"] = new_temp
                        gen_params["do_sample"] = new_temp > 0
                        print(f"  Temperatura: {new_temp}")
                    except ValueError:
                        print("  Error: valor numerico requerido")

            elif cmd == "/tokens":
                if len(parts) < 2:
                    print(f"  Max tokens: {gen_params['max_new_tokens']}")
                else:
                    try:
                        new_tokens = int(parts[1])
                        gen_params["max_new_tokens"] = new_tokens
                        print(f"  Max tokens: {new_tokens}")
                    except ValueError:
                        print("  Error: valor entero requerido")

            elif cmd == "/clear":
                chat_history.clear()
                print("  Historial limpiado")

            else:
                print(f"  Comando desconocido: {cmd}")

            continue

        # Build messages for the task
        task_name = mode_info["task"]

        if current_mode == "chat":
            # Multi-turn: accumulate history
            messages = build_messages(task_name, user_input)
            # Insert history between system and user
            if chat_history:
                system_msg = messages[0]
                user_msg = messages[-1]
                messages = [system_msg] + chat_history + [user_msg]
        else:
            # Single-turn tasks (translation, classification)
            messages = build_messages(task_name, user_input)

        # Generate
        print()
        response = generate_response(
            model,
            tokenizer,
            messages,
            **gen_params,
            stream=True,
        )
        print()

        # Update chat history
        if current_mode == "chat":
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            # Keep history manageable
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GuaraniLM — Interactive Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  translate_gn_es  Translate Guarani to Spanish
  translate_es_gn  Translate Spanish to Guarani
  chat             Bilingual chat
  classify         Sentiment classification

Examples:
  python src/inference.py --model checkpoints/sft/final --mode chat
  python src/inference.py --model skyvanguard/guarani-lm-0.5b --mode translate_gn_es
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (local) or HuggingFace hub ID",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=list(MODES.keys()),
        help="Inference mode (default: chat)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, 0=greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("Loading model: %s", args.model)
    model, tokenizer = load_model(args.model)

    device = next(model.parameters()).device
    logger.info("Model loaded on device: %s", device)

    run_repl(
        model,
        tokenizer,
        mode=args.mode,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

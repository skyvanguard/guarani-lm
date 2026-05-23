"""Quick non-interactive test of the SFT v3 model.

Runs a battery of representative prompts (chat in Guaraní, translation both
directions, culture Q&A) and prints predictions for a qualitative read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/cpt_v3_refine/final")

PROMPTS: list[tuple[str, str, str]] = [
    # (label, task, user_text)
    ("chat-gn-greeting", "chat",
     "Mba'éichapa nde? Eñembo'eke che ñe'ẽmente ha emombe'u chéve mba'éichapa reiko."),
    ("chat-es-asking-gn", "chat",
     "Por favor, contestame en guaraní: ¿qué es el Paraguay para vos?"),
    ("chat-cultura-jopara", "chat",
     "Emombe'u chéve mbykymi mba'épa ha'e Tupã ñande ypy kuéra rembe'ýpe."),
    ("translate-es-gn-1", "translate_es_gn",
     "Buenos días, ¿cómo está tu familia?"),
    ("translate-es-gn-2", "translate_es_gn",
     "El presidente del Paraguay visitó la escuela de Asunción."),
    ("translate-gn-es-1", "translate_gn_es",
     "Mba'éichapa neko'ẽ, che irũ. Aiko porã."),
    ("translate-gn-es-2", "translate_gn_es",
     "Tetãygua paraguái oikuaa mokõi ñe'ẽ: español ha guaraní."),
    ("chat-pregunta-libre", "chat",
     "¿Podés escribirme un poema corto en guaraní sobre el río Paraguay?"),
]


def main() -> None:
    from unsloth import FastLanguageModel
    from prompt_templates import build_messages, format_chatml

    print(f"\n=== Loading model: {MODEL_PATH} ===\n")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model ready. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    for label, task, user_text in PROMPTS:
        messages = build_messages(task, user_text)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = format_chatml(messages)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Per-task generation params:
        # - Translation: greedy (deterministic) + heavy rep penalty + short cap
        # - Chat: low temperature + heavier rep penalty than before
        if task.startswith("translate"):
            gen_kwargs = dict(
                max_new_tokens=80,
                do_sample=False,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                top_k=30,
                repetition_penalty=1.4,
                pad_token_id=tokenizer.pad_token_id,
            )

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen, skip_special_tokens=True).strip()

        print("=" * 70)
        print(f"[{label}] task={task}  params={gen_kwargs}")
        print(f"USER: {user_text}")
        print(f"MODEL: {response}")
        print()


if __name__ == "__main__":
    main()

"""Generate expanded SFT dataset integrating all new datasets.

Adds to the original prepare_instructions.py output:
- Capibara Jopara (50K instruction-following in Jopara)
- Cultura Guaraní Llama-70B (33K QA about Guaraní culture)
- Cultura Guaraní Llama-8B (33K QA about Guaraní culture)
- Cultura Guaraní Corpus (1.4K high-quality QA)
- Alpaca Guaraní Cleaned (52K, replaces original Alpaca)
- Storyteller (3.7K stories -> QA pairs)
- michaelginn/lecslab (1.6K linguistic glosses)

Output: data/processed/sft_v2.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED / "sft_v2.jsonl"

SEED = 42
SYSTEM_PROMPT = "Nde ha'e peteĩ pytyvõhára oñe'ẽva guaraníme ha españolpe."


def make_conv(user_msg: str, assistant_msg: str, task: str, source: str) -> dict:
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ],
        "source": source,
        "task": task,
    }


def load_original_sft() -> list[dict]:
    """Load the original SFT dataset (translations, mmaguero, chat)."""
    path = PROCESSED / "sft_all.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # Keep translations, classification, and chat — skip old alpaca
            if d.get("source") != "alpaca_guarani":
                records.append(d)
    print(f"Original SFT (sin alpaca): {len(records)}")
    return records


def load_capibara_jopara() -> list[dict]:
    """Capibara: instruction/input/output Alpaca format in Jopara."""
    records = []
    for split in ["train", "validation"]:
        path = RAW / "capibara_jopara" / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                instruction = d.get("instruction", "").strip()
                inp = d.get("input", "").strip()
                output = d.get("output", "").strip()
                if not instruction or not output:
                    continue
                user_msg = f"{instruction}\n{inp}" if inp else instruction
                records.append(make_conv(user_msg, output, "instruction_jopara", "capibara_jopara"))
    print(f"Capibara Jopara: {len(records)}")
    return records


def load_alpaca_cleaned() -> list[dict]:
    """Alpaca Guaraní Cleaned: better quality than original."""
    records = []
    path = RAW / "alpaca_guarani_cleaned" / "train.jsonl"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            instruction = d.get("instruction", "").strip()
            inp = d.get("input", "").strip()
            output = d.get("output", "").strip()
            if not instruction or not output:
                continue
            if inp and inp.lower() != "nan":
                user_msg = f"{instruction}\n{inp}"
            else:
                user_msg = instruction
            records.append(make_conv(user_msg, output, "instruction_following", "alpaca_cleaned"))
    print(f"Alpaca Cleaned: {len(records)}")
    return records


def load_cultura_70b() -> list[dict]:
    """Cultura Guaraní Llama-70B: contexto + preguntas (no tiene respuesta directa)."""
    records = []
    path = RAW / "cultura_guarani_llama70b" / "train.jsonl"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            contexto = d.get("contexto", "").strip()
            preguntas = d.get("preguntas", "").strip()
            if not contexto or not preguntas:
                continue
            # Use contexto as the answer (the context IS the knowledge)
            # and preguntas as the question
            records.append(make_conv(preguntas, contexto, "cultura_qa", "cultura_70b"))
    print(f"Cultura 70B: {len(records)}")
    return records


def load_cultura_8b() -> list[dict]:
    """Cultura Guaraní Llama-8B: pregunta + resumen/contexto."""
    records = []
    path = RAW / "cultura_guarani_llama8b" / "train.jsonl"
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pregunta = d.get("pregunta", "").strip()
            resumen = d.get("resumen", "").strip()
            contexto = d.get("contexto", "").strip()
            if not pregunta:
                continue
            # Prefer resumen as answer, fallback to contexto
            answer = resumen if resumen else contexto
            if not answer:
                continue
            records.append(make_conv(pregunta, answer, "cultura_qa", "cultura_8b"))
    print(f"Cultura 8B: {len(records)}")
    return records


def load_cultura_corpus() -> list[dict]:
    """Cultura Guaraní Corpus: preguntas/respuestas (high quality)."""
    records = []
    for split in ["train", "test"]:
        path = RAW / "cultura_guarani_corpus" / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                preguntas = d.get("preguntas", "").strip()
                respuestas = d.get("respuestas", "").strip()
                if not preguntas or not respuestas:
                    continue
                records.append(make_conv(preguntas, respuestas, "cultura_qa", "cultura_corpus"))
    print(f"Cultura Corpus: {len(records)}")
    return records


def load_storyteller() -> list[dict]:
    """Storyteller: seed -> story. Convert to QA about stories."""
    records = []
    path = RAW / "cultura_guarani_storyteller" / "train.jsonl"
    if not path.exists():
        return records

    templates = [
        "Contame una historia sobre: {seed}",
        "Escribí un cuento basado en: {seed}",
        "Emoñe'ẽ peteĩ mombe'u rehegua: {seed}",
        "Ehai peteĩ cuento ko'á rehegua: {seed}",
        "Emombe'u chéve peteĩ historia: {seed}",
    ]

    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            seed = d.get("seed", "").strip()
            story = d.get("story", "").strip()
            if not seed or not story:
                continue
            tmpl = random.choice(templates)
            user_msg = tmpl.format(seed=seed)
            records.append(make_conv(user_msg, story, "storytelling", "storyteller"))
    print(f"Storyteller: {len(records)}")
    return records


def load_linguistic() -> list[dict]:
    """michaelginn/lecslab: transcription + glosses + translation."""
    records = []
    for ds_name in ["michaelginn_guarani", "lecslab_guarani"]:
        path = RAW / ds_name / "train.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                gn = d.get("transcription", "").strip()
                en = d.get("translation", "").strip()
                glosses = d.get("glosses", "").strip()
                if not gn or not en:
                    continue
                # Translation pair gn -> en
                records.append(make_conv(
                    f"Traducí al inglés: {gn}",
                    en,
                    "translate_gn_en", ds_name
                ))
                # Gloss analysis
                if glosses:
                    records.append(make_conv(
                        f"Analizá morfológicamente: {gn}",
                        f"Glosas: {glosses}\nTraducción: {en}",
                        "linguistic_analysis", ds_name
                    ))
    print(f"Linguistic: {len(records)}")
    return records


def main():
    random.seed(SEED)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    all_records = []

    # Original data (translations, mmaguero, chat) — WITHOUT old alpaca
    all_records.extend(load_original_sft())

    # New datasets
    all_records.extend(load_capibara_jopara())
    all_records.extend(load_alpaca_cleaned())
    all_records.extend(load_cultura_70b())
    all_records.extend(load_cultura_8b())
    all_records.extend(load_cultura_corpus())
    all_records.extend(load_storyteller())
    all_records.extend(load_linguistic())

    # Shuffle
    random.shuffle(all_records)

    # Stats
    tasks = Counter(r["task"] for r in all_records)
    sources = Counter(r["source"] for r in all_records)

    print(f"\n=== TOTAL: {len(all_records)} registros ===")
    print("\nPor tarea:")
    for t, c in tasks.most_common():
        print(f"  {t}: {c} ({100*c/len(all_records):.1f}%)")
    print("\nPor fuente:")
    for s, c in sources.most_common():
        print(f"  {s}: {c} ({100*c/len(all_records):.1f}%)")

    # Write
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nEscrito: {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

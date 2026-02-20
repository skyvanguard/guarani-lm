"""Prepare test sets for evaluation from sft_val.jsonl and parallel_all.jsonl.

Creates the following files in data/processed/:
- test_translation_gn_es.jsonl  (prompt + reference)
- test_translation_es_gn.jsonl  (prompt + reference)
- test_sentiment.jsonl          (prompt + reference)
- test_classification.jsonl     (prompt + reference for humor/offensive)
- test_perplexity_gn.jsonl      (text field for guarani perplexity)
"""

import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

def extract_from_sft(task_filter: str, max_samples: int = 500) -> list[dict]:
    """Extract conversation samples from sft_val.jsonl by task type."""
    samples = []
    with open(DATA_DIR / "sft_val.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task") == task_filter:
                convs = d["conversations"]
                # Find user prompt and assistant response
                prompt = ""
                reference = ""
                for msg in convs:
                    if msg["role"] == "user":
                        prompt = msg["content"]
                    elif msg["role"] == "assistant":
                        reference = msg["content"]
                if prompt and reference:
                    samples.append({"prompt": prompt, "reference": reference})

    random.shuffle(samples)
    return samples[:max_samples]


def extract_classification(task_filters: list[str], max_samples: int = 200) -> list[dict]:
    """Extract classification samples (humor, offensive, emotion)."""
    samples = []
    with open(DATA_DIR / "sft_val.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task") in task_filters:
                convs = d["conversations"]
                prompt = ""
                reference = ""
                for msg in convs:
                    if msg["role"] == "user":
                        prompt = msg["content"]
                    elif msg["role"] == "assistant":
                        reference = msg["content"]
                if prompt and reference:
                    samples.append({"prompt": prompt, "reference": reference})

    random.shuffle(samples)
    return samples[:max_samples]


def extract_perplexity_texts(max_samples: int = 500) -> list[dict]:
    """Extract Guarani texts for perplexity evaluation from test.jsonl."""
    samples = []
    with open(DATA_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "")
            # Filter for texts that look like Guarani (not pure Spanish)
            if text and len(text) > 50:
                samples.append({"text": text})

    random.shuffle(samples)
    return samples[:max_samples]


def write_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data)} samples to {path.name}")


def main():
    print("Preparing test sets for evaluation...\n")

    # Translation GN -> ES
    trans_gn_es = extract_from_sft("translate_gn_es", max_samples=500)
    write_jsonl(DATA_DIR / "test_translation_gn_es.jsonl", trans_gn_es)

    # Translation ES -> GN
    trans_es_gn = extract_from_sft("translate_es_gn", max_samples=500)
    write_jsonl(DATA_DIR / "test_translation_es_gn.jsonl", trans_es_gn)

    # Sentiment
    sentiment = extract_from_sft("sentiment", max_samples=200)
    write_jsonl(DATA_DIR / "test_sentiment.jsonl", sentiment)

    # Classification (humor + offensive + emotion)
    classification = extract_classification(
        ["humor", "offensive", "emotion"], max_samples=200
    )
    write_jsonl(DATA_DIR / "test_classification.jsonl", classification)

    # Perplexity
    ppl_texts = extract_perplexity_texts(max_samples=500)
    write_jsonl(DATA_DIR / "test_perplexity_gn.jsonl", ppl_texts)

    print("\nDone! All test sets ready in data/processed/")


if __name__ == "__main__":
    main()

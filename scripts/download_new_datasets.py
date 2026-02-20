"""Download new HuggingFace datasets for GuaraniLM v2 training."""
import json
import os
from datasets import load_dataset

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

DATASETS = {
    "capibara_jopara": "Capibara-LLM/dataset-guarani-jopara-v01",
    "cultura_guarani_llama70b": "thinkPy/synthetic-dataset_cultura-guarani_llama3-70B",
    "cultura_guarani_llama8b": "thinkPy/synthetic-dataset_cultura-guarani_llama3-8B",
    "cultura_guarani_corpus": "somosnlp/dataset-cultura-guarani_corpus-it",
    "alpaca_guarani_cleaned": "saillab/alpaca-guarani-cleaned",
    "cultura_guarani_dpo": "thinkPy/dataset-cultura-guarani_mistral-dpo",
    "cultura_guarani_storyteller": "thinkPy/storyteller_cultura-guarani",
    "cultura_guarani_mistral": "thinkPy/synthetic-dataset_cultura-guarani_mistral-it-v0.3",
    "michaelginn_guarani": "michaelginn/guarani",
    "lecslab_guarani": "lecslab/guarani",
}

total_records = 0

for name, repo in DATASETS.items():
    outdir = os.path.join(BASE, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Descargando {repo}...")
    try:
        ds = load_dataset(repo)
        for split_name, split_data in ds.items():
            outfile = os.path.join(outdir, f"{split_name}.jsonl")
            with open(outfile, "w", encoding="utf-8") as f:
                for row in split_data:
                    f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            print(f"  {split_name}: {len(split_data)} registros -> {outfile}")
            total_records += len(split_data)
            # Show sample fields
            if len(split_data) > 0:
                print(f"  campos: {list(split_data[0].keys())}")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print(f"TOTAL: {total_records} registros nuevos descargados")

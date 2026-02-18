"""Generate instruction-tuning dataset in ChatML format for GuaraniLM SFT.

Takes parallel data and classification data as input, produces ~100K
instruction samples in ChatML message format distributed across task types.

Output: data/processed/instructions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_PARALLEL = PROJECT_ROOT / "data" / "interim" / "parallel_gn_es.jsonl"
DEFAULT_SENTIMENT = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "guarani_sentiment.jsonl"
DEFAULT_HUMOR = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "guarani_humor.jsonl"
DEFAULT_HATE = PROJECT_ROOT / "data" / "raw" / "mmaguero" / "guarani_hate.jsonl"
DEFAULT_AUGMENTED = PROJECT_ROOT / "data" / "interim" / "augmented_nllb.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "instructions.jsonl"

# Target distribution
TARGET_TOTAL = 100_000
DIST = {
    "translate_gn_es": 0.40,
    "translate_es_gn": 0.40,
    "bilingual_chat": 0.05,
    "sentiment": 0.05,
    "classification": 0.055,
    "text_generation": 0.045,
}

# ---------------------------------------------------------------------------
# Instruction templates (5+ per task to avoid overfitting)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = [
    "Nde ha'e peteĩ AI omba'apóva Guaraní ha Español-pe. Eñemoĩ porã ha embohovái hekopete.",
    "Eres un asistente bilingüe Guaraní-Español. Responde de forma precisa y natural.",
    "Nde ha'e peteĩ ñe'ẽryru guaraní-español. Emoĩ porã ne ñe'ẽ.",
    "Eres un modelo de lenguaje experto en Guaraní paraguayo y Jopara.",
    "Nde ha'e peteĩ AI ikatúva oñe'ẽ Guaraní ha Español-pe. Embohováimi porã.",
]

TEMPLATES_GN_ES = [
    "Emoĩ español-pe ko ñe'ẽ: {gn}",
    "Embohasámi ko guaraní-gui español-pe: {gn}",
    "Traduce al español: {gn}",
    "Mba'épa he'ise español-pe: {gn}",
    "Emongueta español-pe: {gn}",
    "Ehai español-pe ko ñe'ẽ: {gn}",
    "Emombe'u español-pe mba'épa he'ise kóva: {gn}",
]

TEMPLATES_ES_GN = [
    "Traduce al guaraní: {es}",
    "Emoĩ guaraní-pe ko ñe'ẽ: {es}",
    "Embohasámi ko español-gui guaraní-pe: {es}",
    "Cómo se dice en guaraní: {es}",
    "Ehai guaraní-pe: {es}",
    "Embohasámi avañe'ẽme: {es}",
    "Emongueta guaraní-pe: {es}",
]

TEMPLATES_BILINGUAL_CHAT = [
    {"user": "Mba'éichapa reime?", "assistant": "Aime porã, aguyje. Ha nde? Mba'épa roipotá?"},
    {"user": "Mba'épa Guaraní?", "assistant": "Guaraní ha'e peteĩ ñe'ẽ oñe'ẽva hetaiterei tapicha Paraguay-pe. Ha'e ñe'ẽ oficial Paraguay retãme, español ndive."},
    {"user": "Contame sobre Paraguay en guaraní.", "assistant": "Paraguay ha'e peteĩ tetã oĩva Sudamérica mbytépe. Oguereko mokõi ñe'ẽ oficial: español ha guaraní. Itáva ha'e Asunción. Paraguay-pe oĩ ysyry guasu: Paraguay ha Paraná."},
    {"user": "Eñe'ẽ chéve yvy ára rehegua.", "assistant": "Ko árape yvy ára iporã. Kuarahy hesãi ha arai mbovy oĩ. Ára pyahu oúma ha yvytu ikyguã."},
    {"user": "Mba'épa jopara?", "assistant": "Jopara ha'e peteĩ ñe'ẽ oñembojoajúva guaraní ha español. Hetaiterei paraguayo oñe'ẽ jopara ko'ẽro ko'ẽro. Ha'e peteĩ ñe'ẽ iporãva ha natural paraguayo-pe guarã."},
    {"user": "Cuáles son las comidas típicas de Paraguay?",  "assistant": "Paraguay-pe comida típica ha'e: sopa paraguaya (torta de maíz y queso), chipa (pan de almidón y queso), mbejú (tortilla de almidón), soyo (sopa de carne), mbeju, vorí vorí (sopa con bolitas de maíz), ha kavure."},
    {"user": "Emombe'u chéve terere rehegua.", "assistant": "Tereré ha'e peteĩ jey'u paraguayo oñembohasáva ka'a ha y ro'ysã ndive. Ha'e peteĩ tradición iporãva Paraguay-pe. Tapichakuéra oñemboja'o tereré okaháre, ha ha'e señal amistad-gua."},
]

TEMPLATES_SENTIMENT = [
    "Mba'e sentimiento oguerekópa ko ñe'ẽ: \"{text}\"",
    "Ehechámi mba'e sentimiento oĩpa ko texto-pe: \"{text}\"",
    "Analiza el sentimiento del siguiente texto en guaraní: \"{text}\"",
    "Mba'épa ko tapicha oñeñandu: \"{text}\"",
    "Es positivo, negativo o neutro: \"{text}\"",
]

SENTIMENT_LABELS = {
    0: "Negativo / Vai",
    1: "Positivo / Iporã",
    2: "Neutro / Mbytépe",
    "negative": "Negativo / Vai",
    "positive": "Positivo / Iporã",
    "neutral": "Neutro / Mbytépe",
    "neg": "Negativo / Vai",
    "pos": "Positivo / Iporã",
    "neu": "Neutro / Mbytépe",
}

TEMPLATES_HUMOR = [
    "Ko texto guaraní-pe, oguerekópa humor: \"{text}\"",
    "Es humorístico este texto: \"{text}\"",
    "Ehechámi oĩpa chiste ko ñe'ẽ-pe: \"{text}\"",
    "Analiza si el siguiente texto en guaraní tiene humor: \"{text}\"",
    "Mba'épa, ko ñe'ẽ opukakavy: \"{text}\"",
]

TEMPLATES_HATE = [
    "Ko texto oguerekópa ñe'ẽ vai: \"{text}\"",
    "Analiza si el siguiente texto contiene discurso de odio: \"{text}\"",
    "Ehechámi oĩpa ofensa ko ñe'ẽ-pe: \"{text}\"",
    "Es ofensivo este texto: \"{text}\"",
    "Mba'épa, ko ñe'ẽ ivaipa: \"{text}\"",
]

CLASSIFICATION_LABELS_BOOL = {
    0: "Nahániri / No",
    1: "Heẽ / Sí",
    False: "Nahániri / No",
    True: "Heẽ / Sí",
    "0": "Nahániri / No",
    "1": "Heẽ / Sí",
}

TEMPLATES_TEXT_GEN = [
    "Ehai peteĩ moñe'ẽrã guaraní-pe {topic} rehegua.",
    "Escribi un párrafo en guaraní sobre {topic}.",
    "Emoĩ peteĩ texto guaraní-pe oñe'ẽva {topic} rehegua.",
    "Ehai peteĩ ñe'ẽ mbykymi guaraní-pe ko tema rehegua: {topic}.",
    "Emongueta guaraní-pe {topic} rehegua. Ehai mokõi térã mbohapy ñe'ẽjuaju.",
]

TEXT_GEN_TOPICS = [
    "Paraguay", "ka'aguy", "ysyry", "tembi'u paraguayo", "aranduka",
    "tereré", "ñe'ẽ guaraní", "Asunción", "ñemba'apo", "tembi'u",
    "mba'e porã", "kuarahy", "yvy", "mainumby", "jagua",
    "oga", "táva", "tembiasakue", "ñe'ẽ", "mitã",
    "arapoty", "yvytu", "ama", "pyhareve", "ka'aru",
    "familia", "mba'éichapa jaiko", "tekojoja", "ñemongeta", "arandu",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts. Returns empty list if missing."""
    if not path.exists():
        print(f"  [warn] Archivo no encontrado: {path}")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def make_chatml(
    system: str, user: str, assistant: str
) -> dict:
    """Create a ChatML-formatted instruction sample."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------


def gen_translate_gn_es(pairs: list[dict], n: int) -> list[dict]:
    """Generate gn->es translation instruction samples."""
    samples = []
    if not pairs:
        return samples

    for _ in range(n):
        pair = random.choice(pairs)
        system = random.choice(SYSTEM_PROMPTS)
        template = random.choice(TEMPLATES_GN_ES)
        user = template.format(gn=pair["gn"])
        assistant = pair["es"]
        samples.append(make_chatml(system, user, assistant))

    return samples


def gen_translate_es_gn(pairs: list[dict], n: int) -> list[dict]:
    """Generate es->gn translation instruction samples."""
    samples = []
    if not pairs:
        return samples

    for _ in range(n):
        pair = random.choice(pairs)
        system = random.choice(SYSTEM_PROMPTS)
        template = random.choice(TEMPLATES_ES_GN)
        user = template.format(es=pair["es"])
        assistant = pair["gn"]
        samples.append(make_chatml(system, user, assistant))

    return samples


def gen_bilingual_chat(n: int) -> list[dict]:
    """Generate bilingual chat instruction samples."""
    samples = []

    for _ in range(n):
        system = random.choice(SYSTEM_PROMPTS)
        convo = random.choice(TEMPLATES_BILINGUAL_CHAT)
        samples.append(make_chatml(system, convo["user"], convo["assistant"]))

    return samples


def gen_sentiment(data: list[dict], n: int) -> list[dict]:
    """Generate sentiment analysis instruction samples."""
    samples = []
    if not data:
        return samples

    for _ in range(n):
        row = random.choice(data)
        text = row.get("text", row.get("sentence", ""))
        label = row.get("label", row.get("sentiment", ""))

        if not text:
            continue

        text = normalize(text)
        label_str = SENTIMENT_LABELS.get(label, str(label))

        system = random.choice(SYSTEM_PROMPTS)
        template = random.choice(TEMPLATES_SENTIMENT)
        user = template.format(text=text)
        assistant = label_str
        samples.append(make_chatml(system, user, assistant))

    return samples


def gen_classification(
    humor_data: list[dict],
    hate_data: list[dict],
    n: int,
) -> list[dict]:
    """Generate classification instruction samples (humor + hate speech)."""
    samples = []

    # Split allocation roughly 50/50
    n_humor = n // 2
    n_hate = n - n_humor

    # Humor detection
    if humor_data:
        for _ in range(n_humor):
            row = random.choice(humor_data)
            text = row.get("text", row.get("sentence", ""))
            label = row.get("label", row.get("is_humor", ""))

            if not text:
                continue

            text = normalize(text)
            label_str = CLASSIFICATION_LABELS_BOOL.get(label, str(label))

            system = random.choice(SYSTEM_PROMPTS)
            template = random.choice(TEMPLATES_HUMOR)
            user = template.format(text=text)
            assistant = label_str
            samples.append(make_chatml(system, user, assistant))

    # Hate speech detection
    if hate_data:
        for _ in range(n_hate):
            row = random.choice(hate_data)
            text = row.get("text", row.get("sentence", ""))
            label = row.get("label", row.get("is_offensive", ""))

            if not text:
                continue

            text = normalize(text)
            label_str = CLASSIFICATION_LABELS_BOOL.get(label, str(label))

            system = random.choice(SYSTEM_PROMPTS)
            template = random.choice(TEMPLATES_HATE)
            user = template.format(text=text)
            assistant = label_str
            samples.append(make_chatml(system, user, assistant))

    return samples


def gen_text_generation(pairs: list[dict], n: int) -> list[dict]:
    """Generate text generation instruction samples in Guarani.

    Uses existing Guarani text as reference outputs.
    """
    samples = []

    for _ in range(n):
        system = random.choice(SYSTEM_PROMPTS)
        topic = random.choice(TEXT_GEN_TOPICS)
        template = random.choice(TEMPLATES_TEXT_GEN)
        user = template.format(topic=topic)

        # Use a real Guarani text as the assistant response if available
        if pairs:
            pair = random.choice(pairs)
            assistant = pair.get("gn", pair.get("text", ""))
        else:
            assistant = f"Ko ha'e peteĩ ñe'ẽ {topic} rehegua."

        samples.append(make_chatml(system, user, assistant))

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera dataset de instrucciones ChatML para GuaraniLM SFT."
    )
    parser.add_argument(
        "--parallel",
        type=Path,
        default=DEFAULT_PARALLEL,
        help="Archivo JSONL de pares paralelos.",
    )
    parser.add_argument(
        "--sentiment",
        type=Path,
        default=DEFAULT_SENTIMENT,
        help="Archivo JSONL de sentimiento.",
    )
    parser.add_argument(
        "--humor",
        type=Path,
        default=DEFAULT_HUMOR,
        help="Archivo JSONL de humor.",
    )
    parser.add_argument(
        "--hate",
        type=Path,
        default=DEFAULT_HATE,
        help="Archivo JSONL de discurso de odio.",
    )
    parser.add_argument(
        "--augmented",
        type=Path,
        default=DEFAULT_AUGMENTED,
        help="Archivo JSONL de datos augmentados via NLLB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Archivo JSONL de salida.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=TARGET_TOTAL,
        help=f"Total de muestras a generar (default: {TARGET_TOTAL}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("GuaraniLM - Generacion de instrucciones ChatML")
    print(f"  Total objetivo: {args.total}")

    # Load data
    print("\nCargando datos ...")
    parallel = load_jsonl(args.parallel)
    augmented = load_jsonl(args.augmented)
    sentiment_data = load_jsonl(args.sentiment)
    humor_data = load_jsonl(args.humor)
    hate_data = load_jsonl(args.hate)

    # Combine parallel sources
    all_pairs = parallel + [r for r in augmented if "gn" in r and "es" in r]

    print(f"  Pares paralelos: {len(parallel)}")
    print(f"  Pares augmentados: {len(augmented)}")
    print(f"  Sentimiento: {len(sentiment_data)}")
    print(f"  Humor: {len(humor_data)}")
    print(f"  Hate speech: {len(hate_data)}")

    if not all_pairs:
        print("\n[error] No se encontraron datos paralelos. Genera primero:")
        print("  python scripts/prepare_parallel.py")
        print("  python scripts/augment_nllb.py  (opcional)")
        sys.exit(1)

    # Calculate target counts per task
    total = args.total
    counts = {task: int(total * frac) for task, frac in DIST.items()}
    # Adjust rounding to hit exact total
    remainder = total - sum(counts.values())
    if remainder > 0:
        counts["translate_gn_es"] += remainder

    print(f"\nDistribucion objetivo:")
    for task, count in counts.items():
        pct = count / total * 100
        print(f"  {task}: {count} ({pct:.1f}%)")

    # Generate samples
    print("\nGenerando muestras ...")
    all_samples: list[dict] = []

    all_samples.extend(gen_translate_gn_es(all_pairs, counts["translate_gn_es"]))
    all_samples.extend(gen_translate_es_gn(all_pairs, counts["translate_es_gn"]))
    all_samples.extend(gen_bilingual_chat(counts["bilingual_chat"]))
    all_samples.extend(gen_sentiment(sentiment_data, counts["sentiment"]))
    all_samples.extend(gen_classification(humor_data, hate_data, counts["classification"]))
    all_samples.extend(gen_text_generation(all_pairs, counts["text_generation"]))

    # Shuffle
    random.shuffle(all_samples)

    print(f"\n  Total generado: {len(all_samples)}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for sample in tqdm(all_samples, desc="Escribiendo"):
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    size_mb = args.output.stat().st_size / (1 << 20)
    print(f"\nGuardado: {args.output} ({len(all_samples)} muestras, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

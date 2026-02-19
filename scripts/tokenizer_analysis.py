"""Analyze Qwen2.5 tokenizer fertility on Guarani text.

Measures how efficiently the Qwen2.5-0.5B tokenizer encodes Guarani
compared to Spanish, by computing tokens-per-word (fertility) statistics.

Prints a summary report to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_GN_INPUT = PROJECT_ROOT / "data" / "raw" / "wikipedia_hf" / "wikipedia_gn.jsonl"
DEFAULT_ES_INPUT = PROJECT_ROOT / "data" / "raw" / "jojajovai_hf" / "jojajovai_hf.jsonl"

# Fallback sample texts if no files available
SAMPLE_GN = [
    "Paraguay ha'e peteĩ tetã oĩva Sudamérica mbytépe.",
    "Guaraní ha'e ñe'ẽ oficial Paraguay retãme.",
    "Ñande jára omoheñói yvága ha yvy.",
    "Ko'ẽro ajúta ne rógape.",
    "Che sy oguata ka'aguy rupi.",
    "Mitãkuña omba'apo mbo'ehaópe.",
    "Tereré ha'e jey'u paraguayo.",
    "Jagua omongyhyje mitãme.",
    "Kuarahy oñembojere yvy ári.",
    "Aranduka tuichaiterei oĩ arandukápe.",
    "Itá guasu oĩ ysyry rembe'ýpe.",
    "Yvytu oipeju hatã ko ka'arúpe.",
    "Mainumby oguejy yvoty ári.",
    "Jasy ohechauka hesa hendýva pyharépe.",
    "Che avy'a aiko porã rupi.",
    "Tembi'u Paraguay-gua iporã ha he.",
    "Sopa paraguaya ha'e tembi'u iporãvéva.",
    "Mbejú ojejapóva almidón ha queso reheve.",
    "Ama ky'a ou ha oho pya'e.",
    "Ka'aguy guasu oĩ Norte Paraguay-pe.",
]

SAMPLE_ES = [
    "Paraguay es un país ubicado en el centro de Sudamérica.",
    "El guaraní es un idioma oficial de Paraguay.",
    "Nuestro creador hizo el cielo y la tierra.",
    "Mañana vendré a tu casa.",
    "Mi madre camina por el bosque.",
    "La niña trabaja en la escuela.",
    "El tereré es una bebida paraguaya.",
    "El perro asustó al niño.",
    "El sol gira sobre la tierra.",
    "Hay una biblioteca muy grande en la ciudad.",
    "Hay una roca grande a la orilla del río.",
    "El viento sopla fuerte esta tarde.",
    "El colibrí baja sobre las flores.",
    "La luna muestra sus ojos brillantes en la noche.",
    "Estoy feliz porque vivo bien.",
    "La comida paraguaya es rica y sabrosa.",
    "La sopa paraguaya es la comida más deliciosa.",
    "El mbejú se hace con almidón y queso.",
    "La lluvia viene y se va rápido.",
    "Hay un gran bosque en el norte de Paraguay.",
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_fertility(
    tokenizer, texts: list[str]
) -> dict:
    """Compute fertility statistics for a list of texts.

    Returns a dict with:
      - mean, median, p25, p75, p95 fertility (tokens/word)
      - total_tokens, total_words
      - word_fertility: dict mapping word -> avg tokens per occurrence
    """
    import statistics

    all_fertilities: list[float] = []
    word_token_counts: Counter = Counter()
    word_occurrences: Counter = Counter()
    total_tokens = 0
    total_words = 0

    for text in texts:
        words = text.split()
        if not words:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        total_words += len(words)

        # Text-level fertility
        fertility = len(tokens) / len(words)
        all_fertilities.append(fertility)

        # Per-word fertility
        for word in words:
            word_clean = word.strip(".,;:!?\"'()-")
            if not word_clean:
                continue
            word_tokens = tokenizer.encode(word_clean, add_special_tokens=False)
            word_token_counts[word_clean] += len(word_tokens)
            word_occurrences[word_clean] += 1

    if not all_fertilities:
        return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p95": 0,
                "total_tokens": 0, "total_words": 0, "word_fertility": {}}

    all_fertilities.sort()
    n = len(all_fertilities)

    # Per-word average fertility
    word_fertility = {}
    for word, total_toks in word_token_counts.items():
        count = word_occurrences[word]
        word_fertility[word] = total_toks / count

    return {
        "mean": statistics.mean(all_fertilities),
        "median": statistics.median(all_fertilities),
        "p25": all_fertilities[max(0, n // 4 - 1)],
        "p75": all_fertilities[max(0, 3 * n // 4 - 1)],
        "p95": all_fertilities[max(0, int(n * 0.95) - 1)],
        "total_tokens": total_tokens,
        "total_words": total_words,
        "word_fertility": word_fertility,
    }


def load_texts_from_jsonl(path: Path, max_texts: int, text_field: str = "text") -> list[str]:
    """Load text strings from a JSONL file."""
    texts: list[str] = []
    if not path.exists():
        return texts

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get(text_field, "")
            except json.JSONDecodeError:
                continue

            if text and len(text) > 20:
                texts.append(text)
            if len(texts) >= max_texts:
                break

    return texts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(
    gn_stats: dict,
    es_stats: dict,
    model_name: str,
    top_n: int = 30,
) -> None:
    """Print a formatted analysis report to stdout."""
    print("=" * 72)
    print(f"  TOKENIZER FERTILITY ANALYSIS - {model_name}")
    print("=" * 72)

    print(f"\n{'Metrica':<30} {'Guarani':>15} {'Espanol':>15} {'Ratio GN/ES':>15}")
    print("-" * 75)

    for label, key in [
        ("Mean fertility (tok/word)", "mean"),
        ("Median fertility", "median"),
        ("P25 fertility", "p25"),
        ("P75 fertility", "p75"),
        ("P95 fertility", "p95"),
    ]:
        gn_val = gn_stats[key]
        es_val = es_stats[key]
        ratio = gn_val / es_val if es_val > 0 else float("inf")
        print(f"{label:<30} {gn_val:>15.3f} {es_val:>15.3f} {ratio:>15.2f}x")

    print("-" * 75)
    gn_total_f = gn_stats["total_tokens"] / max(gn_stats["total_words"], 1)
    es_total_f = es_stats["total_tokens"] / max(es_stats["total_words"], 1)
    print(f"{'Total tokens':<30} {gn_stats['total_tokens']:>15,} {es_stats['total_tokens']:>15,}")
    print(f"{'Total words':<30} {gn_stats['total_words']:>15,} {es_stats['total_words']:>15,}")
    print(f"{'Overall fertility':<30} {gn_total_f:>15.3f} {es_total_f:>15.3f}")

    # Most over-tokenized Guarani words
    gn_word_f = gn_stats.get("word_fertility", {})
    if gn_word_f:
        print(f"\n{'=' * 72}")
        print(f"  TOP {top_n} MOST OVER-TOKENIZED GUARANI WORDS")
        print(f"{'=' * 72}")
        print(f"{'#':<5} {'Word':<30} {'Tokens/word':>15}")
        print("-" * 50)

        sorted_words = sorted(gn_word_f.items(), key=lambda x: x[1], reverse=True)
        for i, (word, fertility) in enumerate(sorted_words[:top_n], 1):
            print(f"{i:<5} {word:<30} {fertility:>15.1f}")

    # Words well-handled (fertility <= 1.5)
    well_handled = [(w, f) for w, f in gn_word_f.items() if f <= 1.5]
    if well_handled:
        print(f"\n  Palabras bien tokenizadas (fertility <= 1.5): {len(well_handled)}")
        print(f"  Palabras sobre-tokenizadas (fertility > 2.0): "
              f"{len([w for w, f in gn_word_f.items() if f > 2.0])}")
        print(f"  Total palabras unicas analizadas: {len(gn_word_f)}")

    print(f"\n{'=' * 72}")
    print("  INTERPRETACION")
    print(f"{'=' * 72}")
    ratio = gn_total_f / es_total_f if es_total_f > 0 else 0
    if ratio > 2.0:
        print(f"  El tokenizer es {ratio:.1f}x MENOS eficiente en Guarani que en Espanol.")
        print("  RECOMENDACION: Considerar extender el vocabulario del tokenizer con")
        print("  tokens frecuentes en Guarani (nasal vowels, puso, suffixes).")
    elif ratio > 1.5:
        print(f"  El tokenizer es {ratio:.1f}x menos eficiente en Guarani que en Espanol.")
        print("  Esto es esperado para un modelo multilingue. La diferencia es moderada.")
    else:
        print(f"  El tokenizer tiene eficiencia similar en Guarani y Espanol ({ratio:.1f}x).")
        print("  Esto es un buen indicador para el fine-tuning.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza la fertilidad del tokenizer Qwen2.5 en Guarani vs Espanol."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Modelo/tokenizer a analizar (default: Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--gn-input",
        type=Path,
        default=DEFAULT_GN_INPUT,
        help="Archivo JSONL con textos en guaraní.",
    )
    parser.add_argument(
        "--es-input",
        type=Path,
        default=DEFAULT_ES_INPUT,
        help="Archivo JSONL con textos en español.",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=1000,
        help="Maximo de textos a analizar por idioma (default: 1000).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Cantidad de palabras mas sobre-tokenizadas a mostrar (default: 30).",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Usar textos de muestra incorporados (no requiere archivos).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"GuaraniLM - Analisis de tokenizer")
    print(f"  Modelo: {args.model}")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[error] 'transformers' no instalado. Ejecuta: pip install transformers")
        sys.exit(1)

    print(f"  Cargando tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Vocabulario: {tokenizer.vocab_size:,} tokens")

    # Load texts
    if args.use_samples:
        gn_texts = SAMPLE_GN
        es_texts = SAMPLE_ES
        print(f"  Usando textos de muestra incorporados")
    else:
        gn_texts = load_texts_from_jsonl(args.gn_input, args.max_texts, text_field="text")
        es_texts = load_texts_from_jsonl(args.es_input, args.max_texts, text_field="es")

        if not gn_texts:
            print(f"  [info] Sin textos GN en {args.gn_input}, usando muestras incorporadas")
            gn_texts = SAMPLE_GN
        if not es_texts:
            print(f"  [info] Sin textos ES en {args.es_input}, usando muestras incorporadas")
            es_texts = SAMPLE_ES

    print(f"  Textos Guarani: {len(gn_texts)}")
    print(f"  Textos Espanol: {len(es_texts)}")

    # Compute fertility
    print(f"\n  Analizando fertilidad ...")
    gn_stats = compute_fertility(tokenizer, gn_texts)
    es_stats = compute_fertility(tokenizer, es_texts)

    # Print report
    print_report(gn_stats, es_stats, args.model, top_n=args.top_n)


if __name__ == "__main__":
    main()

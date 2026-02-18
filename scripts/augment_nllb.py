"""Data augmentation using NLLB-200 for Guarani LM training.

Uses facebook/nllb-200-distilled-600M for:
  - Forward translation: es -> gn (from Spanish text sources)
  - Back-translation: gn -> es -> gn (for quality filtering)

Output: data/interim/augmented_nllb.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from normalize_guarani import normalize

PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_SPANISH_INPUT = PROJECT_ROOT / "data" / "raw" / "spanish_snippets.jsonl"
DEFAULT_GUARANI_INPUT = PROJECT_ROOT / "data" / "interim" / "parallel_gn_es.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "augmented_nllb.jsonl"

# NLLB-200 language codes
LANG_GN = "grn_Latn"  # Guarani
LANG_ES = "spa_Latn"  # Spanish

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Default confidence threshold for filtering
MIN_SCORE = 0.5


# ---------------------------------------------------------------------------
# Translation engine
# ---------------------------------------------------------------------------


class NLLBTranslator:
    """Wrapper around NLLB-200 for batch translation."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto") -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
        except ImportError:
            print("[error] 'transformers' y 'torch' son necesarios.")
            print("  pip install transformers torch sentencepiece")
            sys.exit(1)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Cargando modelo {model_name} en {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        print(f"  Modelo cargado en {device}.")

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        max_length: int = 256,
    ) -> list[tuple[str, float]]:
        """Translate a batch of texts, returning (translation, score) pairs.

        The score is the mean log-probability of the generated tokens,
        converted to a 0-1 range via exp().
        """
        import torch

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=max_length,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode translations
        translations = self.tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True
        )

        # Compute approximate confidence scores from sequence scores
        scores: list[float] = []
        if hasattr(generated, "sequences_scores") and generated.sequences_scores is not None:
            for seq_score in generated.sequences_scores:
                # sequences_scores are log-probs; convert to 0-1 range
                score = torch.exp(seq_score).item()
                scores.append(min(score, 1.0))
        else:
            # Fallback: no scores available
            scores = [1.0] * len(translations)

        return list(zip(translations, scores))


# ---------------------------------------------------------------------------
# Forward translation (es -> gn)
# ---------------------------------------------------------------------------


def forward_translate(
    translator: NLLBTranslator,
    spanish_texts: list[str],
    batch_size: int,
    min_score: float,
) -> list[dict]:
    """Translate Spanish texts to Guarani (forward translation)."""
    results: list[dict] = []

    print(f"\nForward translation (es->gn): {len(spanish_texts)} textos")

    for i in tqdm(range(0, len(spanish_texts), batch_size), desc="Forward es->gn"):
        batch = spanish_texts[i : i + batch_size]
        translated = translator.translate_batch(batch, LANG_ES, LANG_GN)

        for es_text, (gn_text, score) in zip(batch, translated):
            if score < min_score:
                continue

            gn_text = normalize(gn_text)
            if len(gn_text.strip()) < 10:
                continue

            results.append({
                "gn": gn_text,
                "es": es_text,
                "source": "nllb_forward",
                "score": round(score, 4),
            })

    return results


# ---------------------------------------------------------------------------
# Back-translation (gn -> es -> gn)
# ---------------------------------------------------------------------------


def back_translate(
    translator: NLLBTranslator,
    guarani_texts: list[str],
    batch_size: int,
    min_score: float,
) -> list[dict]:
    """Back-translate Guarani texts (gn -> es -> gn) for quality filtering."""
    results: list[dict] = []

    print(f"\nBack-translation (gn->es->gn): {len(guarani_texts)} textos")

    for i in tqdm(range(0, len(guarani_texts), batch_size), desc="Back-translation"):
        batch_gn = guarani_texts[i : i + batch_size]

        # Step 1: gn -> es
        gn_to_es = translator.translate_batch(batch_gn, LANG_GN, LANG_ES)
        es_texts = [t for t, _ in gn_to_es]
        fwd_scores = [s for _, s in gn_to_es]

        # Step 2: es -> gn (back)
        es_to_gn = translator.translate_batch(es_texts, LANG_ES, LANG_GN)

        for orig_gn, es_text, (bt_gn, bt_score), fwd_score in zip(
            batch_gn, es_texts, es_to_gn, fwd_scores
        ):
            # Combined score: geometric mean of forward and backward scores
            combined_score = (fwd_score * bt_score) ** 0.5

            if combined_score < min_score:
                continue

            bt_gn = normalize(bt_gn)
            if len(bt_gn.strip()) < 10:
                continue

            results.append({
                "gn": bt_gn,
                "es": es_text,
                "gn_original": orig_gn,
                "source": "nllb_backtranslation",
                "score": round(combined_score, 4),
            })

    return results


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_spanish_texts(path: Path, max_samples: int) -> list[str]:
    """Load Spanish text snippets for forward translation."""
    texts: list[str] = []

    if not path.exists():
        print(f"  [warn] Archivo de textos en español no encontrado: {path}")
        print("  Se pueden generar con fuentes externas (Wikipedia ES, noticias, etc.)")
        return texts

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("text", record.get("es", ""))
            except json.JSONDecodeError:
                text = line

            if text and 20 <= len(text) <= 500:
                texts.append(text)

            if len(texts) >= max_samples:
                break

    return texts


def load_guarani_texts(path: Path, max_samples: int) -> list[str]:
    """Load Guarani texts for back-translation."""
    texts: list[str] = []

    if not path.exists():
        print(f"  [warn] Archivo de textos en guaraní no encontrado: {path}")
        return texts

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("gn", record.get("text", ""))
            except json.JSONDecodeError:
                continue

            if text and 20 <= len(text) <= 500:
                texts.append(text)

            if len(texts) >= max_samples:
                break

    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aumenta datos Guarani usando NLLB-200 (traduccion forward y back-translation)."
    )
    parser.add_argument(
        "--spanish-input",
        type=Path,
        default=DEFAULT_SPANISH_INPUT,
        help="Archivo con textos en español para forward translation.",
    )
    parser.add_argument(
        "--guarani-input",
        type=Path,
        default=DEFAULT_GUARANI_INPUT,
        help="Archivo con textos en guaraní para back-translation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Archivo JSONL de salida.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño de batch para traduccion (default: 16).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Maximo de muestras por fuente (default: 10000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo para inferencia (default: auto).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help=f"Score minimo de confianza para filtrado (default: {MIN_SCORE}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Modelo NLLB a usar (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Saltar forward translation (solo back-translation).",
    )
    parser.add_argument(
        "--skip-backtranslation",
        action="store_true",
        help="Saltar back-translation (solo forward).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("GuaraniLM - Augmentacion con NLLB-200")
    print(f"  Modelo: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Min score: {args.min_score}")

    # Initialize translator
    translator = NLLBTranslator(model_name=args.model, device=args.device)

    all_results: list[dict] = []
    start_time = time.time()

    # Forward translation (es -> gn)
    if not args.skip_forward:
        spanish_texts = load_spanish_texts(args.spanish_input, args.max_samples)
        if spanish_texts:
            print(f"\n  {len(spanish_texts)} textos en español cargados")
            fwd_results = forward_translate(
                translator, spanish_texts, args.batch_size, args.min_score
            )
            all_results.extend(fwd_results)
            print(f"  Forward: {len(fwd_results)} pares generados")
        else:
            print("\n  [info] Sin textos en español, saltando forward translation")

    # Back-translation (gn -> es -> gn)
    if not args.skip_backtranslation:
        guarani_texts = load_guarani_texts(args.guarani_input, args.max_samples)
        if guarani_texts:
            print(f"\n  {len(guarani_texts)} textos en guaraní cargados")
            bt_results = back_translate(
                translator, guarani_texts, args.batch_size, args.min_score
            )
            all_results.extend(bt_results)
            print(f"  Back-translation: {len(bt_results)} pares generados")
        else:
            print("\n  [info] Sin textos en guaraní, saltando back-translation")

    elapsed = time.time() - start_time

    # Write results
    if all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fout:
            for record in all_results:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        size_mb = args.output.stat().st_size / (1 << 20)
        print(f"\nGuardado: {args.output} ({len(all_results)} pares, {size_mb:.1f} MB)")
    else:
        print("\n[warn] No se generaron resultados.")

    print(f"Tiempo total: {elapsed:.1f}s")

    # Score distribution
    if all_results:
        scores = [r["score"] for r in all_results]
        scores.sort()
        n = len(scores)
        print(f"\nDistribucion de scores:")
        print(f"  Min: {scores[0]:.4f}")
        print(f"  P25: {scores[n // 4]:.4f}")
        print(f"  P50: {scores[n // 2]:.4f}")
        print(f"  P75: {scores[3 * n // 4]:.4f}")
        print(f"  Max: {scores[-1]:.4f}")


if __name__ == "__main__":
    main()

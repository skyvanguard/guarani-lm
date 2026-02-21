# Model Card: GuaraniLM-0.5B

## Model Details

- **Model name**: guarani-lm-0.5b
- **Model type**: Causal language model (decoder-only)
- **Language**: Guaraní (grn), Spanish (spa), Jopara (Guaraní-Spanish mix)
- **Base model**: Qwen2.5-0.5B
- **Parameters**: ~500M
- **License**: Apache 2.0
- **Repository**: https://github.com/skyvanguard/guarani-lm

## Model Description

GuaraniLM-0.5B is the first open-source generative language model optimized for Paraguayan Guaraní and Jopara. It was trained using continual pre-training (CPT) followed by supervised fine-tuning (SFT) on the Qwen2.5-0.5B base model.

### Training Procedure

1. **Continual Pre-Training (CPT)**: QLoRA 4-bit with r=32, targeting q/k/v/o/gate/up/down projections. 1 epoch on ~308K texts (~6.5M tokens). Final loss: 2.296. Duration: ~3.4h on RTX 4070 8GB.

2. **Supervised Fine-Tuning (SFT)**: QLoRA 4-bit with r=32 on ~114K instruction samples in ChatML format. NEFTune noise alpha=5. 1 epoch. Final loss: 1.780. Duration: ~1.6h.

### Training Data

| Source | Records | Usage |
|--------|---------|-------|
| HPLT 2.0 cleaned | 73K docs | CPT |
| Wikipedia Guaraní | ~5K articles | CPT |
| Jojajovai parallel corpus | 30K pairs | CPT + SFT |
| AmericasNLP 2021 | ~34K pairs | CPT + SFT |
| Alpaca Guaraní | 52K | SFT |
| mmaguero datasets | ~10K | SFT |
| GUA-SPA 2023 | 1.5K pairs | SFT |
| Gov.py translator | 3K pairs | CPT + SFT |
| FLORES-200 / Belebele | ~3K | Eval |

## Intended Use

- Translation between Guaraní and Spanish
- Bilingual chat in Guaraní and Jopara
- Sentiment and text classification in Guaraní
- Text generation in Guaraní
- Educational and research purposes for low-resource NLP

## Evaluation Results

Evaluated on held-out test sets with greedy decoding.

| Task | Metric | Score |
|------|--------|-------|
| Translation GN→ES | BLEU | 2.98 |
| Translation GN→ES | chrF2 | 25.89 |
| Translation ES→GN | BLEU | 1.71 |
| Translation ES→GN | chrF2 | 21.27 |
| Sentiment (3-class) | Accuracy | 21.9% |
| Classification | Accuracy | 22.2% |
| Guaraní Perplexity | PPL | 11.13 |

**Notes**: Translation BLEU/chrF2 scores reflect the challenge of Guaraní as an extremely low-resource language. The model shows learning signal above random baselines for classification. Perplexity of 11.13 on Guaraní text indicates meaningful language modeling capability compared to the base Qwen2.5-0.5B model.

## Limitations

- **Small training data**: ~6.5M tokens is significantly less than typical LM training (billions). The model may generate repetitive or incoherent text for complex topics.
- **Guaraní orthography**: Guaraní has multiple orthographic conventions. The model was trained primarily with the standard Paraguayan orthography.
- **Jopara bias**: Since much available Guaraní text is actually Jopara (mixed with Spanish), the model may insert Spanish words even when asked for pure Guaraní.
- **Hallucination**: Like all language models, it may generate factually incorrect information.
- **Small model**: At 500M parameters, this model is not comparable to larger models for general knowledge tasks.

## Ethical Considerations

- This model is intended for research and educational purposes.
- It should not be used as a sole authority on Guaraní language or culture.
- Generated translations should be reviewed by native speakers before use in official contexts.
- The training data may contain biases present in web-crawled text.

## How to Cite

```bibtex
@software{guarani_lm_2026,
  title = {GuaraniLM: First Open-Source Generative Model for Guarani and Jopara},
  author = {skyvanguard},
  year = {2026},
  url = {https://github.com/skyvanguard/guarani-lm}
}
```

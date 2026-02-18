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

1. **Continual Pre-Training**: QLoRA 4-bit with r=128, targeting all linear layers + embeddings. Trained on ~6.5M tokens of Guaraní text from Wikipedia, CulturaX, Jojajovai parallel corpus, and NLLB-200 augmented data.

2. **Supervised Fine-Tuning**: LoRA r=64 on ~100K instruction samples covering translation, chat, classification, and text generation in ChatML format.

### Training Data

| Source | Tokens | Usage |
|--------|--------|-------|
| Wikipedia Guaraní | ~1.5M | CPT |
| CulturaX (grn) | ~1-3M | CPT |
| Jojajovai parallel corpus | ~1.2M | CPT + SFT |
| mmaguero datasets | ~300K | SFT |
| NLLB-200 augmentation | ~2M | CPT + SFT |

## Intended Use

- Translation between Guaraní and Spanish
- Bilingual chat in Guaraní and Jopara
- Sentiment and text classification in Guaraní
- Text generation in Guaraní
- Educational and research purposes for low-resource NLP

## Evaluation Results

| Task | Metric | Baseline | GuaraniLM |
|------|--------|----------|-----------|
| Translation GN→ES | chrF2 | NLLB-200 | TBD |
| Translation ES→GN | chrF2 | NLLB-200 | TBD |
| Sentiment | Macro-F1 | gn-bert | TBD |
| Perplexity GN | PPL | Qwen2.5-0.5B | TBD |

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

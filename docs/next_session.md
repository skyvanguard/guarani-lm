# Próxima sesión — Single-command resume

Cuando tengas ~16h GPU libre, ejecutá esto. No requiere ninguna preparación.

## Lanzar SFT round 2

```bash
cd C:\Users\skyva\Documents\guarani-lm
docker compose run --rm trainer bash -c "python3 src/train_sft.py --config configs/sft_v3_round2_config.yaml 2>&1 | tee logs/sft_v3_round2.log"
```

ETA: ~16h. Checkpoints cada 250 steps en `checkpoints/sft_v3_round2/checkpoint-*`.
Para correr en background y desconectarte:

```bash
docker compose run -d --rm trainer bash -c "python3 src/train_sft.py --config configs/sft_v3_round2_config.yaml 2>&1 > logs/sft_v3_round2.log"
```

## Después del training (cuando termine)

### Re-exportar a GGUF
```bash
docker compose run --rm trainer python3 scripts/export_gguf.py \
    --model checkpoints/sft_v3_round2/final \
    --output checkpoints/sft_v3_round2/gguf
```

### Re-crear modelo Ollama
Antes editá `Modelfile.v3` línea 14 (`FROM`) para apuntar al nuevo GGUF:
```
FROM ./checkpoints/sft_v3_round2/gguf/model-q4_k_m.gguf
```

Después:
```bash
ollama create guarani-lm-v3 -f Modelfile.v3
ollama run guarani-lm-v3
```

### Re-correr eval para comparar
Editá `configs/eval_v3_config.yaml` línea 4 a `checkpoints/sft_v3_round2/final`, después:
```bash
docker compose run --rm trainer python3 src/evaluate.py --config configs/eval_v3_config.yaml
```

Comparar contra `eval/results/eval_results_20260521_011708.json` (resultado v3 cpt_refine).

## Estado actual del proyecto (snapshot)

| Artefacto | Path | Tamaño | Notas |
|---|---|---|---|
| Modelo actual (escribe guaraní, no sigue instrucciones) | `checkpoints/cpt_v3_refine/final/` | 5.8 GB | Listo para inferencia |
| GGUF actual | `checkpoints/cpt_v3_refine/gguf/model-q4_k_m.gguf` | 1.8 GB | Cargado en Ollama como `guarani-lm-v3` |
| Datos CPT limpios | `data/processed/cpt_train_clean.jsonl` | 211 MB | 157K records (de 308K originales) |
| Datos SFT limpios | `data/processed/sft_v2_train_clean.jsonl` | 182 MB | 236K records (99.99% kept) |
| Config próximo training | `configs/sft_v3_round2_config.yaml` | — | LR=5e-6, 1 epoch, data clean |

## Qué esperar del SFT round 2

**Objetivo**: recuperar instruction-following sin perder la calidad lingüística del CPT.

Métricas que deberían subir vs cpt_v3_refine actual:
- sentiment accuracy: 0.25 → esperaría 0.40+
- classification accuracy: 0.14 → esperaría 0.30+
- BLEU es→gn: 1.20 → esperaría > 1.5
- Tokens basura (URLs, hashtags, emojis en respuestas) → casi 0

Métricas que deberían mantenerse:
- perplexity: 8.56 (no debería subir mucho)
- BLEU gn→es: 2.46
- Capacidad de escribir poesía guaraní

Si el resultado es bueno, el modelo final estará en `checkpoints/sft_v3_round2/final/`.

## Notas

- El Modelfile.v3 ya tiene params calibrados (temp=0.6, rep_penalty=1.8) que
  funcionan bien post-CPT. Probablemente no necesiten ajuste después de round 2.
- Si el dataset limpio resultó demasiado agresivo (sentís que algo importante
  se perdió), podés revertir al original cambiando el config a `sft_v2_train.jsonl`.
- Los originales sucios siguen en disco como respaldo.

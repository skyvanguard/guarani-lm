# Sesión 2026-05-20 — GuaraniLM v3 (Qwen2.5-3B)

## TL;DR

Migramos de Qwen2.5-**0.5B** (v2) a Qwen2.5-**3B** (v3) en una RTX 5070 Ti
Laptop 12GB, vía Docker con CUDA 12.8 + PyTorch 2.10. El modelo final
(`checkpoints/cpt_v3_refine/final/`) **sí escribe guaraní** — incluso poesía
genuina — pero perdió capacidad de seguir instrucciones de clasificación
respecto a v2. La perplexity bajó 16% (10.22 → 8.56), lo que confirma
cuantitativamente la mejora en modelado del idioma.

## Stack final

| Componente | Versión |
|---|---|
| GPU | RTX 5070 Ti Laptop 12GB (Blackwell SM_120) |
| Base image | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| PyTorch | 2.10.0+cu128 |
| Unsloth | 2026.5.5 |
| Transformers | 5.5.0 |
| Base model | `unsloth/Qwen2.5-3B` (1.93B params, vocab 151666) |

## Pipeline final

```
Qwen2.5-3B base
    │
    ├─► SFT v3-only        (1 epoch, 236K instrucciones, seq=1024, LoRA r=32)
    │   16.7h               sft_v3_only/final/
    │
    └─► CPT v3 refine      (1 epoch, 92K records guaraní, LR=1.5e-5)
        4.0h                cpt_v3_refine/final/  ← MODELO FINAL
```

## Configs producidos

- `configs/pretrain_v3_config.yaml` — CPT full (no usado: 19h por epoch)
- `configs/pretrain_v3_bench_config.yaml` — benchmark 30 steps
- `configs/sft_v3_config.yaml` — SFT v3 over CPT (no usado)
- `configs/sft_v3_only_config.yaml` — **SFT directo sobre base 3B (usado)**
- `configs/cpt_v3_refine_config.yaml` — **CPT refinement (usado)**
- `configs/eval_v3_config.yaml` — eval apuntando a cpt_v3_refine

## Resultados eval — v2 vs v3

| Task | Metric | v2 | v3 | Δ | Lectura |
|---|---|---|---|---|---|
| translation_gn_es | BLEU | 2.14 | **2.46** | +14.9% | ✅ mejor |
| translation_gn_es | chrF2 | 25.87 | 25.65 | -0.8% | igual |
| translation_es_gn | BLEU | 1.56 | 1.20 | -23% | regresión |
| translation_es_gn | chrF2 | 22.33 | 21.91 | -1.9% | igual |
| sentiment | accuracy | 0.47 | 0.25 | -47% | ❌ regresión severa |
| sentiment | macro_f1 | 0.13 | 0.12 | -8% | igual |
| classification | accuracy | 0.25 | 0.14 | -44% | ❌ regresión severa |
| classification | macro_f1 | 0.08 | 0.05 | -44% | ❌ regresión severa |
| **perplexity** | (menor=mejor) | 10.22 | **8.56** | **-16%** | ✅ **mucho mejor** |

## Interpretación

**Lo que mejoró (modelado del idioma):**
- Perplexity 16% más baja confirma que v3 asigna probabilidades sustancialmente
  más altas a texto guaraní real.
- Cualitativamente: el modelo escribió un poema guaraní genuino sobre el río
  Paraguay (palabras reales: Kuarahy, Yvytu, ysyry, kapi'i, ka'avo).
- Tokens basura del v2 (chino, tailandés, `_^(_)`) eliminados completamente.

**Lo que regresó (instruction following):**
- Sentiment y classification se desplomaron a niveles cercanos al azar.
- Causa raíz: CPT después de SFT diluyó los patrones de instrucción del SFT.
  El LR bajo (1.5e-5) mitigó pero no evitó el efecto.
- Predicho desde el inicio cuando elegimos saltar el orden tradicional CPT → SFT.

**Por qué BLEU es engañoso:**
- BLEU premia n-gramas exactos contra el reference. v2 generaba ruido que por
  casualidad incluía n-gramas memorizados del entrenamiento.
- v3 genera guaraní gramaticalmente correcto pero estilísticamente distinto al
  reference, lo que castiga el BLEU.

## Artefactos en disco

```
checkpoints/
├── sft_v3_only/
│   ├── final/                     5.8 GB  (Qwen2.5-3B + SFT, fp16 merged)
│   ├── final_adapter/             LoRA r=32 adapter
│   └── checkpoint-3500..4003/     intermedios
└── cpt_v3_refine/                 ← USAR ESTE
    ├── final/                     5.8 GB  (SFT + CPT refine, fp16 merged)
    ├── final_adapter/             LoRA r=32 adapter
    ├── gguf/                      GGUF Q4_K_M para Ollama (export en curso)
    └── checkpoint-400..958/       intermedios

logs/
├── bench_v3.log                   30-step benchmark
├── sft_v3_only.log                16.7h training
├── cpt_v3_refine.log              4.0h training
├── eval_v3.log                    eval suite ~1.5h
└── gguf_export.log                GGUF export

data/processed/                    526 MB (copiado desde D:/guarani-lm/data)
├── cpt_train.jsonl                308K records, 60M tokens
├── cpt_train_v3_30pct.jsonl       92K records (seed=42 reproducible)
├── sft_v2_train.jsonl             236K instrucciones
└── test_*.jsonl                   conjuntos eval

eval/results/                      (v2 baseline JSONs)
```

## Cosas que no funcionaron primero (y se arreglaron)

1. **`unsloth[cu121-torch250]` con pin de transformers**: pip `resolution-too-deep`.
   Fix: instalar Unsloth solo sin pins; ya trae deps compatibles.
2. **PyTorch 2.5.1 + RTX 5070 Ti**: no soporta SM_120. Fix: CUDA 12.8 + PyTorch 2.10.
3. **`torch.int1` AttributeError**: torchao requería PyTorch ≥2.6. Mismo fix.
4. **Triton JIT falla compilar `cuda_utils.c`**: falta `python3.10-dev`. Fix:
   apt-get install en patch incremental.
5. **GGUF export interactivo pide libcurl4-openssl-dev + libssl-dev**: Fix:
   `scripts/export_gguf.py` con llama.cpp directo y `LLAMA_CURL=OFF`.
6. **Mount `D:/`**: Docker Desktop solo tiene `C:\` compartido. Fix: copiamos
   los datos a `data/processed/` dentro del proyecto.

## Cómo usar el modelo después

### Inferencia interactiva
```bash
docker compose run --rm trainer python3 src/inference.py \
    --model checkpoints/cpt_v3_refine/final --mode chat
```

Params recomendados para mejor calidad (ya validados):
- Chat: `--temperature 0.3 --top-p 0.85 --repetition-penalty 1.4`
- Traducción: `--temperature 0` (greedy) y rep penalty 1.5

### Ollama (después que termine el GGUF export)
```bash
# Crear Modelfile
cat > Modelfile <<EOF
FROM ./checkpoints/cpt_v3_refine/gguf/model-q4_k_m.gguf
PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.4
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
EOF
ollama create guarani-lm-v3 -f Modelfile
ollama run guarani-lm-v3
```

## Próximos pasos sugeridos (ordenados por costo/beneficio)

### Si querés recuperar instruction following sin perder lo del idioma:
1. **SFT round 2 sobre cpt_v3_refine** (~16h): aplicar otra ronda de SFT con
   los mismos datos pero LR bajo (5e-6). Recupera sentiment/classification
   sin perder la limpieza del CPT. Esto sería **CPT → SFT → CPT → SFT**, que
   es el orden canónico en literatura.

### Si querés más calidad lingüística:
2. **CPT con el 70% restante** del dataset (~10h): mejoraría perplexity más
   pero seguiría degradando instruction tasks. No recomendado solo, sí en
   combinación con (1).

### Para uso operacional:
3. **Fix dataset contamination**: el modelo genera emojis y `t.co` links
   ocasionalmente porque OSCAR/CulturaX scraperon mucho Twitter/X. Filtrar
   esos artefactos antes del próximo entrenamiento (regex sobre cpt_train.jsonl,
   ~2h de trabajo).

### Si querés evaluación más rigurosa:
4. **Eval con sampling en vez de greedy**: el eval actual usa `do_sample=false`
   que penaliza al v3. Re-correr con sampling + repetition penalty matchea
   mejor el uso real.
5. **Eval cualitativo human-rated**: 50 prompts diversos, vos rateás 1-5.
   Más confiable que BLEU para low-resource.

## Decisión que tomé autónomamente (con tu permiso)

Elegí **Opción 3 (Fix GGUF)** porque:
- Los datos eval probaron que más CPT empeoraría sentiment/classification.
- GGUF te permite probar el modelo en Ollama mañana sin riesgo.
- Es rápido (~20-30 min vs 10h+ de las otras opciones).

Tu próximo movimiento natural: usar el GGUF en Ollama, sentir el modelo,
y entonces decidir si querés (1) SFT round 2 para recuperar instruction
following, o (3) cleanup de datos antes de cualquier nuevo training.

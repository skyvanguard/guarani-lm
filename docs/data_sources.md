# GuaraniLM — Inventario de Datos y Viabilidad de Entrenamiento

Fecha: 2026-02-18
Hardware objetivo: RTX 4070 Laptop (8GB VRAM), 16GB RAM, Intel Ultra 7 155H

---

## 1. Inventario completo de fuentes de datos

### 1.1 Texto monolingue (para Continual Pre-Training)

| # | Fuente | Registros | Tokens est. | Formato | Licencia | Calidad | URL |
|---|--------|-----------|-------------|---------|----------|---------|-----|
| 1 | HPLT 2.0 cleaned | 73,420 docs | ~40M | Parquet | CC0 | Media-Baja | https://hplt-project.org/datasets/v2.0 |
| 2 | HPLT 2.0 dedup | 3.02M docs | ~1.64B palabras | Parquet | CC0 | Baja (ruido) | https://hplt-project.org/datasets/v2.0 |
| 3 | Wikipedia GN (Vikipeta) | ~4,000-5,000 arts | ~800K-1.5M | XML dump | CC BY-SA | Alta | https://dumps.wikimedia.org/gnwiki/ |
| 4 | CC-100 (Facebook) | ? | ~500K | TXT.xz (~1.5MB) | CC terms | Media | https://data.statmt.org/cc-100/ |
| 5 | Leipzig Corpora | 14,612 oraciones | 174K | TXT | Academica | Media | https://corpora.uni-leipzig.de/en?corpusId=grn_community_2017 |
| 6 | CulturaX (grn) | 103 docs | 12,708 | Parquet | CC BY-SA | Baja | https://huggingface.co/datasets/uonlp/CulturaX |
| 7 | OSCAR 23.01 | 14 docs | 260 | Parquet | CC0 | Inutilizable | https://huggingface.co/datasets/oscar-corpus/OSCAR-2301 |

**Notas:**
- HPLT 2.0 es la fuente dominante. La version "cleaned" (227 MB) tiene filtros de calidad aplicados. La version "dedup" (7.21 GB) es masiva pero con mucho ruido.
- CulturaX y OSCAR son despreciables para Guarani.
- Wikipedia es pequena pero de alta calidad (curada por humanos).

### 1.2 Texto paralelo (para CPT + SFT)

| # | Fuente | Pares | Tokens est. | Formato | Licencia | Calidad | URL |
|---|--------|-------|-------------|---------|----------|---------|-----|
| 8 | Jojajovai | 30,855 | ~1.2M | CSV | MIT | Alta | https://github.com/pln-fing-udelar/jojajovai |
| 9 | AmericasNLP 2021 | 33,938 | ~1.3M | TSV | Academica | Alta | https://github.com/AmericasNLP/americasnlp2021 |
| 10 | JW300 (OPUS) | ~100K est. | ~3-4M | TMX/Moses | Varia | Media | https://opus.nlpl.eu/ |
| 11 | NLLB-Seed | 6,193 | ~60K | TXT | CC BY-SA | Muy Alta | https://github.com/facebookresearch/flores |
| 12 | Tatoeba | ~3,367 | ~40K | TSV | CC BY 2.0 | Media | https://tatoeba.org |
| 13 | Gongora corpus | noticias+tweets | ~100K | Varios | Academica | Alta | https://github.com/sgongora27/giossa-gongora-guarani-2021 |

**Notas:**
- Jojajovai es el corpus paralelo de mayor calidad, con anotaciones dialectales por hablantes nativos.
- AmericasNLP incluye datos de shared task, posible overlap con Jojajovai (verificar).
- JW300 es texto religioso (Watchtower/Awake), repetitivo pero voluminoso.
- NLLB-Seed son traducciones profesionales de muy alta calidad.

### 1.3 Datasets de clasificacion/SFT

| # | Fuente | Registros | Tokens est. | Formato | Licencia | Calidad | URL |
|---|--------|-----------|-------------|---------|----------|---------|-----|
| 14 | Alpaca-Guarani | 52,000 | ~2M | Parquet | CC BY-NC | Baja (Google Translate) | https://huggingface.co/datasets/saillab/alpaca-guarani-cleaned |
| 15 | mmaguero sentiment | 5,020 | ~75K | Parquet | No espec. | Alta* | https://huggingface.co/datasets/mmaguero/gn-jopara-sentiment-analysis |
| 16 | mmaguero offensive | 2,170 | ~30K | Parquet | No espec. | Alta* | https://huggingface.co/datasets/mmaguero/gn-offensive-language-identification |
| 17 | mmaguero humor | 1,840 | ~25K | Parquet | No espec. | Alta* | https://huggingface.co/datasets/mmaguero/gn-humor-detection |
| 18 | mmaguero emotion | 1,570 | ~22K | Parquet | No espec. | Alta* | https://huggingface.co/datasets/mmaguero/gn-emotion-recognition |
| 19 | GUA-SPA 2023 | 1,500 | ~25K | Parquet | Academica | Alta | https://huggingface.co/datasets/mmaguero/gua-spa-2023-task-1-2 |
| 20 | Cultura Guarani QA | 1,498 | ~50K | Parquet | CC BY-SA 4.0 | Media | https://huggingface.co/datasets/somosnlp/dataset-cultura-guarani_corpus-it |

*Los datasets mmaguero contienen tweet IDs, NO texto. Requieren X/Twitter API (de pago) para rehidratar. Muchos tweets ya eliminados.

**Notas:**
- Alpaca-Guarani es la traduccion automatica (Google Translate) del dataset Alpaca-52K de Stanford. Calidad baja pero util como base para SFT si se filtra.
- Cultura Guarani QA esta en ESPANOL (no Guarani) — es QA sobre cultura Guarani.

### 1.4 Diccionarios y recursos lexicos

| # | Fuente | Entries | Tokens est. | Formato | Licencia | Calidad | URL |
|---|--------|---------|-------------|---------|----------|---------|-----|
| 21 | Paraguayologia diccionario | 1,300+ | ~100K | Web (scraping) | ? | Alta | https://paraguayologia.com/diccionario-paraguayo/ |
| 22 | Paraguayologia traductor | 6,937 pares | ~50K | Web (scraping) | ? | Alta | https://paraguayologia.com/traductor-guarani-espanol/ |
| 23 | Glosbe GN-ES | 6,992 frases + 149K ejemplos | ~500K-1M | Web | ToS restric. | Media | https://es.glosbe.com/gn/es |

### 1.5 Fuentes por scrapear (no descargadas aun)

| # | Fuente | Estimacion | Tipo | Calidad | URL |
|---|--------|-----------|------|---------|-----|
| 24 | Orembae (Biblioteca Virtual) | 14,000+ poemas/escritos, ~2-5M tokens | Monolingue GN | Alta | https://www.orembae.org.py |
| 25 | ABC Color Remiandy | Miles de articulos | Paralelo GN-ES | Alta | https://www.abc.com.py/especiales/remiandu/ |

### 1.6 Augmentation sintetico

| # | Fuente | Potencial | Tokens est. | Calidad | URL |
|---|--------|-----------|-------------|---------|-----|
| 26 | NLLB-200 distilled 600M | Forward + back-translation | ~2-5M | Media | https://huggingface.co/facebook/nllb-200-distilled-600M |
| 27 | Grammar augmentation (Baladon) | Oraciones sinteticas por gramatica | ~1M | Baja | https://github.com/AlexisBaladon/SyntaxGrammar-es-gn |

### 1.7 Benchmarks de evaluacion (NO para training)

| # | Fuente | Registros | Uso | URL |
|---|--------|-----------|-----|-----|
| 28 | FLORES-200 | 2,009 oraciones | Eval traduccion | https://huggingface.co/datasets/facebook/flores |
| 29 | Belebele | ~900 QA | Eval comprension | https://huggingface.co/datasets/facebook/belebele |
| 30 | NLLB-Seed | 6,193 pares | Eval traduccion | https://github.com/facebookresearch/flores |

### 1.8 Fuentes descartadas

| Fuente | Razon |
|--------|-------|
| mC4 (108 idiomas) | Guarani (gn) NO esta incluido |
| OSCAR 23.01 | Solo 14 docs, 260 palabras — inutilizable |
| CulturaX | Solo 103 docs, 12,708 tokens — despreciable |
| Bible corpus (christos-c) | Guarani no confirmado en el repo |
| Common Voice | Audio/ASR, no texto (33 hrs) |

---

## 2. Estimacion total de tokens

### Escenario conservador (solo fuentes confirmadas, sin scraping)

| Categoria | Tokens |
|-----------|--------|
| HPLT 2.0 cleaned (filtrado 50%) | ~20M |
| Wikipedia GN | ~1M |
| CC-100 + Leipzig | ~700K |
| Jojajovai + AmericasNLP | ~2.5M |
| JW300 | ~3M |
| Alpaca-Guarani (filtrado 30%) | ~600K |
| mmaguero + GUA-SPA | ~150K |
| NLLB augmentation | ~2M |
| **TOTAL conservador** | **~30M** |

### Escenario optimista (con scraping + augmentation agresiva)

| Categoria | Tokens |
|-----------|--------|
| HPLT 2.0 cleaned (filtrado 30%) | ~28M |
| Wikipedia GN | ~1.5M |
| CC-100 + Leipzig | ~700K |
| Jojajovai + AmericasNLP | ~2.5M |
| JW300 | ~3M |
| Orembae scraping | ~3M |
| ABC Color Remiandy | ~2M |
| Paraguayologia | ~150K |
| Alpaca-Guarani (filtrado 50%) | ~1M |
| NLLB augmentation | ~5M |
| Grammar augmentation | ~1M |
| **TOTAL optimista** | **~48M** |

### Comparacion con proyectos existentes

| Proyecto | Tokens training | Params | Resultado |
|----------|----------------|--------|-----------|
| gn-bert (mmaguero) | 800K | 110M (BERT) | Mediocre |
| GuaraniLM conservador | ~30M | 500M (Qwen) | **37x mas datos** |
| GuaraniLM optimista | ~48M | 500M (Qwen) | **60x mas datos** |
| TinyLlama (referencia) | 3T | 1.1B | Estado del arte |

---

## 3. Viabilidad en la notebook

### 3.1 Especificaciones del hardware

```
GPU:  NVIDIA GeForce RTX 4070 Laptop
VRAM: 8 GB GDDR6
CPU:  Intel Core Ultra 7 155H (16 cores / 22 threads)
RAM:  16 GB DDR5
SSD:  1 TB NVMe
OS:   Windows 11 / WSL2 Ubuntu 22.04
```

### 3.2 Requerimientos de VRAM por fase

#### Fase 1: Descarga y procesamiento de datos
- **VRAM**: 0 GB (CPU only)
- **RAM**: 4-8 GB (para HPLT 2.0 el parquet de 227 MB cabe en RAM)
- **Disco**: ~5-10 GB para datos raw + processed
- **NLLB augmentation**: ~2.5 GB VRAM (modelo 600M en fp16)
- **Tiempo**: 4-8 horas (descarga + procesamiento + augmentation)
- **VIABLE**: Si

#### Fase 2: Continual Pre-Training (CPT)

Configuracion: Qwen2.5-0.5B, QLoRA 4-bit, r=128

```
Modelo base 4-bit:           ~350 MB
LoRA adapters r=128:         ~400 MB (todos los modulos + embed + lm_head)
Optimizer states (AdamW):    ~800 MB (2x LoRA params en fp32)
Activaciones (batch=4):      ~800 MB
Gradient checkpointing:      ahorra ~40% activaciones
KV cache (seq_len=2048):     ~200 MB
Overhead PyTorch/CUDA:       ~500 MB
─────────────────────────────────────
TOTAL estimado:              ~2.6-3.5 GB
```

Con **30M tokens**, 3 epochs, batch efectivo 32:
- Steps: (30M / 2048) * 3 / 32 = ~1,373 steps
- Velocidad estimada: ~2-3 steps/sec en RTX 4070
- **Tiempo: ~8-12 minutos** (sorprendentemente rapido por el modelo pequeno)
- **VRAM: ~3-3.5 GB** (bien dentro de los 8 GB)
- **VIABLE**: Si, con margen amplio

**Riesgo**: si r=128 es muy pesado, reducir a r=64 baja VRAM ~200 MB.

#### Fase 3: Supervised Fine-Tuning (SFT)

Configuracion: Checkpoint CPT, LoRA r=64, sin embed_tokens/lm_head

```
Modelo base 4-bit:           ~350 MB
LoRA adapters r=64:          ~150 MB (solo linear layers)
Optimizer states:            ~300 MB
Activaciones (batch=4):      ~800 MB
Gradient checkpointing:      ahorra ~40%
KV cache:                    ~200 MB
Overhead:                    ~500 MB
─────────────────────────────────────
TOTAL estimado:              ~2.0-2.5 GB
```

Con ~100K instrucciones, 3 epochs, batch efectivo 16:
- Steps: ~18,750
- Velocidad: ~2-3 steps/sec
- **Tiempo: ~1.5-2.5 horas**
- **VRAM: ~2-2.5 GB**
- **VIABLE**: Si, facilmente

#### Fase 4: Evaluacion

- Inferencia de modelo 4-bit: ~500 MB VRAM
- Baseline NLLB-200 (600M): ~2.5 GB VRAM
- **Ambos caben** en 8 GB sin problemas
- **VIABLE**: Si

#### Fase 5: Export GGUF

- Merge LoRA + export: pico de ~4 GB RAM (CPU, no GPU)
- Conversion a GGUF: llama.cpp en CPU
- **VIABLE**: Si

### 3.3 Cuello de botella: NLLB augmentation

El unico paso que puede ser lento es correr NLLB-200 para augmentation:
- Modelo: 600M params, fp16 = ~1.2 GB VRAM
- Batch size 8, seq_len 128: ~2.5 GB VRAM total
- Velocidad: ~50-100 traducciones/min en RTX 4070
- Para 20,000 traducciones: ~3-7 horas
- **VIABLE**: Si, pero lento. Considerar batch size 16 si cabe.

### 3.4 Resumen de viabilidad

| Fase | VRAM pico | RAM pico | Disco | Tiempo | Viable? |
|------|-----------|----------|-------|--------|---------|
| Descarga datos | 0 | 4 GB | 10 GB | 2-4h | Si |
| Procesamiento | 0 | 8 GB | 5 GB | 1-2h | Si |
| NLLB augmentation | 2.5 GB | 4 GB | 1 GB | 3-7h | Si |
| CPT (QLoRA r=128) | 3-3.5 GB | 8 GB | 2 GB | 8-12 min | Si |
| SFT (LoRA r=64) | 2-2.5 GB | 6 GB | 1 GB | 1.5-2.5h | Si |
| Evaluacion | 2.5 GB | 4 GB | - | 30 min | Si |
| Export GGUF | 0 (CPU) | 4 GB | 2 GB | 10 min | Si |
| **TOTAL** | **3.5 GB max** | **8 GB max** | **~20 GB** | **~10-18h** | **Si** |

**Conclusion: TODO cabe en la RTX 4070 con 8GB VRAM, con margen.**

El modelo Qwen2.5-0.5B es lo suficientemente pequeno que QLoRA 4-bit usa solo ~3.5 GB en el pico. Esto deja ~4.5 GB libres para el SO y otros procesos.

### 3.5 Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigacion |
|--------|-------------|-----------|
| bitsandbytes falla en Windows | Alta | Usar WSL2 (Ubuntu 22.04 ya instalado) |
| HPLT 2.0 tiene mucho ruido | Media | Filtrar agresivamente (idioma, calidad, dedup) |
| Unsloth incompatible | Baja | Fallback a PEFT + transformers directo |
| OOM en CPT r=128 | Baja (~3.5GB) | Reducir a r=64, batch=2, gradient_accum=16 |
| NLLB traducciones malas | Media | Back-translation cruzada + filtrado por score |
| Alpaca-Guarani inutilizable | Alta | Solo usar 10-20% mejor, descartar resto |
| Overlap Jojajovai/AmericasNLP | Media | Deduplicar por hash antes de merge |

### 3.6 Alternativa: entrenar en WSL2

Dado que bitsandbytes no funciona nativo en Windows, el entrenamiento se hara en WSL2:

```bash
# WSL2 ya configurado: memory=14GB, swap=32GB
# CUDA toolkit accesible via NVIDIA driver de Windows

wsl
cd /mnt/c/Users/skyva/Documents/github-contributions/guarani-lm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

WSL2 accede a la misma GPU RTX 4070 via el driver NVIDIA de Windows. No hay penalidad de rendimiento significativa.

---

## 4. Plan de implementacion detallado

### Semana 1: Descarga y validacion de datos

```
Dia 1-2: Descargar fuentes principales
  - HPLT 2.0 cleaned (227 MB parquet)
  - Wikipedia GN dump
  - Jojajovai (HF o GitHub)
  - AmericasNLP shared task data
  - CC-100, Leipzig, Tatoeba
  - JW300 via OPUS
  - Alpaca-Guarani
  - mmaguero datasets (4)
  - GUA-SPA 2023
  - NLLB-Seed, FLORES-200, Belebele

Dia 3: Validar calidad
  - Muestrear 100 textos de HPLT 2.0, verificar que son Guarani real
  - Contar tokens reales de cada fuente
  - Detectar overlap entre Jojajovai y AmericasNLP
  - Evaluar Alpaca-Guarani: muestrear 50, clasificar calidad 1-5
```

### Semana 2: Scraping + procesamiento

```
Dia 4-5: Scraping
  - Orembae: explorar estructura, scrapear poemas/escritos en GN
  - Paraguayologia: scrapear diccionario (1,300+ entries)
  - ABC Color Remiandy: scrapear articulos en Guarani
  - Verificar legalidad/ToS de cada sitio

Dia 6-7: Pipeline de procesamiento
  - normalize_guarani.py sobre todos los textos
  - clean_wikipedia.py
  - Filtrar HPLT 2.0 (eliminar no-Guarani, dedup, calidad)
  - prepare_parallel.py (Jojajovai + AmericasNLP + JW300)
  - Deduplicacion global
```

### Semana 3: Augmentation + merge

```
Dia 8-9: NLLB augmentation
  - Forward translation es->gn (10-20K oraciones selectas)
  - Back-translation para filtrado
  - Filtrar por score de confianza

Dia 10: Merge final
  - merge_datasets.py: combinar todo
  - Split 90/5/5
  - Reporte de tokens finales
  - tokenizer_analysis.py: verificar fertility < 4
```

### Semana 4: Training

```
Dia 11: CPT
  - python src/train_cpt.py (estimado: 10-15 min)
  - Verificar loss curve, no diverge
  - Guardar checkpoint

Dia 12: SFT
  - Preparar instrucciones ChatML (~100K)
  - python src/train_sft.py (estimado: 2-3h)
  - Probar traduccion interactiva

Dia 13: Evaluacion
  - python src/evaluate.py
  - Comparar vs NLLB-200 zero-shot
  - Comparar vs Qwen2.5-0.5B base
```

### Semana 5: Publicacion

```
Dia 14: Export y publicacion
  - Merge LoRA adapters
  - Subir a HuggingFace: skyvanguard/guarani-lm-0.5b
  - Convertir a GGUF (Q4_K_M, Q8_0)
  - Subir dataset: skyvanguard/guarani-instructions
  - Model card con metricas reales
```

---

## 5. Decision: que fuentes usar y cuales no

### USAR (prioridad alta)

| Fuente | Razon |
|--------|-------|
| HPLT 2.0 cleaned | Mayor volumen por lejos, filtrar bien |
| Wikipedia GN | Alta calidad, texto curado |
| Jojajovai | Mejor corpus paralelo disponible |
| AmericasNLP | Alta calidad, complementa Jojajovai |
| CC-100 | Complemento monolingue |
| Leipzig | Complemento monolingue |
| NLLB augmentation | Escalar datos paralelos |
| FLORES/Belebele/NLLB-Seed | Evaluacion |

### USAR CON CAUTELA

| Fuente | Razon de cautela |
|--------|-----------------|
| JW300 | Texto religioso repetitivo, puede sesgar el modelo |
| Alpaca-Guarani | Google Translate, filtrar agresivamente |
| HPLT 2.0 dedup (7 GB) | Demasiado ruido, solo si cleaned es insuficiente |

### SCRAPEAR SI ES LEGAL

| Fuente | Potencial |
|--------|-----------|
| Orembae | 14K+ poemas, muy valioso si accesible |
| Paraguayologia | Diccionario rico en contexto cultural |
| ABC Color Remiandy | Noticias en Guarani puro |

### NO USAR

| Fuente | Razon |
|--------|-------|
| CulturaX (103 docs) | Despreciable |
| OSCAR (14 docs) | Inutilizable |
| Glosbe | ToS prohiben scraping |
| mmaguero tweets (IDs) | API de X/Twitter de pago, tweets borrados |
| Cultura Guarani QA | Esta en espanol, no Guarani |

---

## 6. Metricas de exito

| Metrica | Objetivo minimo | Objetivo ideal |
|---------|----------------|----------------|
| Tokens de entrenamiento totales | >15M | >40M |
| chrF2 traduccion GN->ES | > NLLB-200 zero-shot | +5 puntos sobre NLLB |
| chrF2 traduccion ES->GN | > NLLB-200 zero-shot | +5 puntos sobre NLLB |
| Perplexidad GN | < Qwen2.5-0.5B base | <50% de base |
| Macro-F1 sentimiento | > gn-bert (si datos disponibles) | >0.7 |
| Fertility tokenizer | < 4 tokens/palabra | < 3 |
| VRAM pico entrenamiento | < 8 GB | < 5 GB |
| Tiempo total entrenamiento | < 24h | < 6h |

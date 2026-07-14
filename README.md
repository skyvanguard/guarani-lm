# GuaraniLM

**Primer modelo generativo de codigo abierto para Guarani y Jopara**

**Peteiha modelo generativo open-source Guarani ha Jopara-pe guarã**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-skyvanguard%2Fguarani--lm--0.5b-yellow)](https://huggingface.co/skyvanguard/guarani-lm-0.5b)

---

## Que es GuaraniLM? / Mba'epa GuaraniLM?

El Guarani es hablado por mas de 7 millones de personas en Paraguay, pero tiene casi cero soporte en modelos de lenguaje modernos. **GuaraniLM** es el primer modelo generativo (decoder-only) de codigo abierto diseñado para Guarani paraguayo y Jopara (mezcla Guarani-Español).

Guarani he'i 7 millon tapicha Paraguay-pe, ha upeicharõ jepe ndaipori modelo de lenguaje iporã Guarani-pe guarã. **GuaraniLM** ha'e peteiha modelo generativo open-source oñembohérava Guarani paraguayo ha Jopara-pe guarã.

### Caracteristicas principales

| Feature | GuaraniLM | gn-bert (existente) |
|---------|-----------|---------------------|
| Tipo | Generativo (decoder-only) | Encoder-only |
| Tareas | Chat, traduccion, generacion | Solo clasificacion |
| Tokens de entrenamiento | ~6.5M (v1/v2) / ~31.8M (v3) | ~800K |
| Formato | HuggingFace + GGUF (Ollama) | Solo PyTorch |
| Base | Qwen2.5-0.5B (v1/v2) / Qwen2.5-3B (v3) | BERT multilingual |

### Generaciones del modelo

| Version | Base | Corpus | Estado |
|---------|------|--------|--------|
| v1 | Qwen2.5-0.5B | ~6.5M tokens, CPT 1 epoch + SFT 114K | Completado |
| v2 | Qwen2.5-0.5B | ~6.5M tokens, CPT 3 epochs + SFT 249K | Completado (release actual) |
| v3 | Qwen2.5-3B | ~31.8M tokens (34,764 docs), pipeline Docker | En progreso (SFT round 2 pendiente) |

### Tareas soportadas

- **Traduccion** Guarani <-> Español
- **Chat bilingüe** en Guarani y Jopara
- **Clasificacion** de sentimiento, humor, ofensividad
- **Generacion de texto** en Guarani

---

## Instalacion

```bash
# Clonar el repo
git clone https://github.com/skyvanguard/guarani-lm.git
cd guarani-lm

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias base
pip install -e .

# Para entrenamiento (GPU requerida)
pip install -e ".[train]"

# Para desarrollo
pip install -e ".[dev]"
```

## Uso rapido

### Con Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "skyvanguard/guarani-lm-0.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "Emombe'u chéve Paraguay rehegua."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Con Ollama

```bash
ollama run skyvanguard/guarani-lm
>>> Mba'éichapa reime?
```

---

## Pipeline de datos

### Corpus v1/v2 (~6.5M tokens)

| Fuente | Tokens | Uso |
|--------|--------|-----|
| Wikipedia Guarani (~5K articulos) | ~1.5M | Pre-training |
| CulturaX subset `grn` | ~1-3M | Pre-training |
| Jojajovai (30K pares gn<->es) | ~1.2M | Pre-training + SFT |
| mmaguero datasets | ~300K | SFT |
| NLLB-200 augmentation | ~2M | Pre-training + SFT |
| **Total** | **~6.5M** | |

### Corpus v3 (~31.8M tokens, 34,764 docs, 127M chars)

Recolectado con 8+ scrapers propios (ver `scripts/`):

- Corpora de HuggingFace: HPLT 2.0, mOSCAR, FineTranslations
- Biblias en Guarani (NNP2015, GPC2006, GDC2006)
- Publicaciones WOL JW (12 categorias, 8K+ docs)
- Oremba'e (pipeline OCR con PyMuPDF + Tesseract)
- Articulos de IP Gov Paraguay y noticias ABC Remiandu
- Filtrado de idioma con fasttext + heuristicas de Guarani, deduplicacion MD5

Para v3 el corpus ademas paso por limpieza de contaminacion (`scripts/clean_contamination.py`): cpt_train se redujo de 308K a 157K registros eliminando URLs, handles y hashtags, preservando 95.5% del contenido.

## Entrenamiento

El entrenamiento se realiza en dos fases: **Continual Pre-Training (CPT)** con QLoRA 4-bit y **Supervised Fine-Tuning (SFT)** con instrucciones en formato ChatML.

### v1/v2 (Qwen2.5-0.5B, GPU >=8GB VRAM)

```bash
# v1: fases separadas
python src/train_cpt.py --config configs/pretrain_config.yaml
python src/train_sft.py --config configs/sft_config.yaml

# v2: pipeline completo (CPT 3 epochs -> SFT 249K x2 epochs, ~19h)
bash run_training_v2.sh
```

### v3 (Qwen2.5-3B, Docker, RTX 5070 Ti 12GB / Blackwell)

v3 corre dentro de un contenedor con CUDA 12.8 + PyTorch 2.10 + Unsloth (soporte SM_120):

```bash
# Construir la imagen y lanzar el pipeline
docker compose run --rm trainer bash run_training_v3.sh
```

Pipeline ejecutado hasta ahora: SFT directo sobre la base 3B (16.7h) -> refinamiento CPT (4h) -> limpieza de dataset. Queda pendiente el SFT round 2 sobre datos limpios (`configs/sft_v3_round2_config.yaml`, ~16h).

## Evaluacion

Resultados en test sets held-out con greedy decoding (ver `docs/model_card.md` para detalles):

| Tarea | Metrica | v1 | v2 | v3 (intermedio) |
|-------|---------|----|----|-----------------|
| Traduccion GN->ES | BLEU | 2.98 | 2.14 | 2.46 |
| Traduccion GN->ES | chrF2 | 25.89 | 25.87 | - |
| Traduccion ES->GN | BLEU | 1.71 | 1.56 | 1.20 |
| Traduccion ES->GN | chrF2 | 21.27 | 22.34 | - |
| Sentimiento (3 clases) | Accuracy | 21.9% | **46.9%** | 25%* |
| Clasificacion | Accuracy | 22.2% | 24.6% | 14%* |
| Perplexidad GN | PPL | 11.13 | 10.22 | **8.56** |

\* El checkpoint v3 evaluado es intermedio (CPT despues de SFT degrada el instruction-following, como se esperaba). El SFT round 2 pendiente busca recuperar esas metricas manteniendo la ganancia de -16% en perplexidad.

```bash
python src/evaluate.py --config configs/eval_v2_config.yaml  # v2
python src/evaluate.py --config configs/eval_v3_config.yaml  # v3
```

---

## Estructura del proyecto

```
guarani-lm/
├── configs/          # Configuraciones YAML para training/eval (v1, v2, v3)
├── scripts/          # Pipeline de datos: descarga, scraping, limpieza, preparacion
├── src/              # Codigo principal: training, evaluacion, inferencia
├── docker/           # Imagen CUDA 12.8 para entrenamiento v3 (Blackwell)
├── notebooks/        # Exploracion de datos y analisis
├── eval/             # Benchmarks y resultados
├── tests/            # Tests unitarios
├── docs/             # Documentacion y model card
├── Modelfile.v3      # Modelfile de Ollama con params calibrados para v3
└── run_training*.sh  # Orquestadores de pipeline por generacion
```

## Contribuir

Las contribuciones son bienvenidas. En particular necesitamos ayuda con:

- Mas datos en Guarani (textos, traducciones, conversaciones)
- Evaluacion humana de las traducciones
- Pruebas con hablantes nativos de Guarani
- Documentacion en Guarani

## Licencia

Apache 2.0. Ver [LICENSE](LICENSE).

## Citar

```bibtex
@software{guarani_lm_2026,
  title = {GuaraniLM: First Open-Source Generative Model for Guarani and Jopara},
  author = {skyvanguard},
  year = {2026},
  url = {https://github.com/skyvanguard/guarani-lm}
}
```

---

*Aguyje opavave omba'apovape ko proyecto-pe*

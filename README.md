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
| Tokens de entrenamiento | ~6.5M | ~800K |
| Formato | HuggingFace + GGUF (Ollama) | Solo PyTorch |
| Base | Qwen2.5-0.5B | BERT multilingual |

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

| Fuente | Tokens | Uso |
|--------|--------|-----|
| Wikipedia Guarani (~5K articulos) | ~1.5M | Pre-training |
| CulturaX subset `grn` | ~1-3M | Pre-training |
| Jojajovai (30K pares gn<->es) | ~1.2M | Pre-training + SFT |
| mmaguero datasets | ~300K | SFT |
| NLLB-200 augmentation | ~2M | Pre-training + SFT |
| **Total** | **~6.5M** | |

## Entrenamiento

El entrenamiento se realiza en dos fases:

1. **Continual Pre-Training (CPT)**: QLoRA 4-bit sobre Qwen2.5-0.5B con r=128, entrenando embeddings + todos los linear layers
2. **Supervised Fine-Tuning (SFT)**: LoRA r=64 sobre el checkpoint CPT, con ~100K instrucciones en formato ChatML

**Requisitos**: GPU con >=8GB VRAM (RTX 3070/4070 o superior)

```bash
# Fase 1: Continual Pre-Training (~2-4h)
python src/train_cpt.py --config configs/pretrain_config.yaml

# Fase 2: Instruction Fine-Tuning (~6-10h)
python src/train_sft.py --config configs/sft_config.yaml
```

## Evaluacion

| Tarea | Metrica | Baseline | GuaraniLM |
|-------|---------|----------|-----------|
| Traduccion GN->ES | chrF2 | NLLB-200 | TBD |
| Traduccion ES->GN | chrF2 | NLLB-200 | TBD |
| Clasificacion sentimiento | Macro-F1 | gn-bert | TBD |
| Perplexidad GN | PPL | Qwen2.5-0.5B | TBD |

```bash
python src/evaluate.py --config configs/eval_config.yaml
```

---

## Estructura del proyecto

```
guarani-lm/
├── configs/          # Configuraciones YAML para training/eval
├── scripts/          # Pipeline de datos: descarga, limpieza, preparacion
├── src/              # Codigo principal: training, evaluacion, inferencia
├── notebooks/        # Exploracion de datos y analisis
├── eval/             # Benchmarks y resultados
├── tests/            # Tests unitarios
└── docs/             # Documentacion y model card
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

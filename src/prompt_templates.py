"""ChatML prompt templates for GuaraniLM tasks.

Provides diverse prompt templates for translation, chat, sentiment analysis,
classification, and text generation.  Templates vary in phrasing and language
(Spanish / Guarani) to improve instruction-following generalization.
"""

from __future__ import annotations

import random
from typing import Any


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

CHATML_TEMPLATE = (
    "<|im_start|>{role}\n{content}<|im_end|>\n"
)


def format_chatml(messages: list[dict[str, str]]) -> str:
    """Format a list of message dicts into a ChatML string.

    Each message must have ``role`` and ``content`` keys.  An
    ``<|im_start|>assistant`` generation prompt is appended at the end.

    Parameters
    ----------
    messages : list[dict[str, str]]
        Conversation turns, e.g.
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``

    Returns
    -------
    str
        The formatted ChatML string ready for tokenization.
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(CHATML_TEMPLATE.format(role=msg["role"], content=msg["content"]))
    # Add generation prompt
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ============================================================================
# Template definitions
# ============================================================================

# ---------------------------------------------------------------------------
# Translation Guarani -> Spanish
# ---------------------------------------------------------------------------

TEMPLATES_TRANSLATE_GN_ES: list[dict[str, str]] = [
    {
        "system": "Eres un traductor experto de Guaraní a Español.",
        "user": "Traducí al español: {source}",
    },
    {
        "system": "Eres un traductor profesional especializado en Guaraní paraguayo.",
        "user": "¿Cómo se dice en español? {source}",
    },
    {
        "system": "Sos un asistente bilingüe Guaraní-Español.",
        "user": "Pasá esto al español por favor:\n{source}",
    },
    {
        "system": "Traducí fielmente del Guaraní al Español.",
        "user": "Texto en Guaraní:\n{source}\n\nTraducción al Español:",
    },
    {
        "system": "Nde ha'e peteĩ traductor Guaraní-Español.",
        "user": "Emonguatia castellano-pe ko texto:\n{source}",
    },
    {
        "system": "Eres un modelo de lenguaje bilingüe entrenado para traducir entre Guaraní y Español.",
        "user": "Traducción Guaraní → Español:\nGuaraní: {source}\nEspañol:",
    },
    {
        "system": "Asistente de traducción Guaraní-Español. Traducí de forma natural y precisa.",
        "user": "{source}\n\nTraducí al castellano.",
    },
]

# ---------------------------------------------------------------------------
# Translation Spanish -> Guarani
# ---------------------------------------------------------------------------

TEMPLATES_TRANSLATE_ES_GN: list[dict[str, str]] = [
    {
        "system": "Eres un traductor experto de Español a Guaraní.",
        "user": "Traducí al guaraní: {source}",
    },
    {
        "system": "Nde ha'e peteĩ traductor Castellano-gui Guaraní-pe.",
        "user": "Emonguatia Guaraní-pe: {source}",
    },
    {
        "system": "Sos un asistente bilingüe Guaraní-Español.",
        "user": "¿Cómo se dice en guaraní? {source}",
    },
    {
        "system": "Traducí fielmente del Español al Guaraní paraguayo.",
        "user": "Texto en Español:\n{source}\n\nTraducción al Guaraní:",
    },
    {
        "system": "Eres un modelo de lenguaje bilingüe entrenado para traducir entre Español y Guaraní.",
        "user": "Traducción Español → Guaraní:\nEspañol: {source}\nGuaraní:",
    },
    {
        "system": "Asistente de traducción Español-Guaraní. Traducí de forma natural.",
        "user": "{source}\n\nTraducí al guaraní.",
    },
    {
        "system": "Emonguatia porã Guaraní-pe ko texto castellano.",
        "user": "Castellano: {source}\nGuaraní:",
    },
]

# ---------------------------------------------------------------------------
# Bilingual chat
# ---------------------------------------------------------------------------

TEMPLATES_CHAT: list[dict[str, str]] = [
    {
        "system": "Sos un asistente bilingüe que habla Guaraní y Español. Respondé en el idioma que te hablen.",
        "user": "{source}",
    },
    {
        "system": "Nde ha'e peteĩ asistente omba'apóva Guaraní ha Castellano-pe. Embohova pe ñe'ẽ ojeporúvape.",
        "user": "{source}",
    },
    {
        "system": "Eres un asistente virtual paraguayo que entiende Guaraní, Español y Jopara.",
        "user": "{source}",
    },
    {
        "system": "Asistente bilingüe paraguayo. Hablás Guaraní, Castellano y Jopara con naturalidad.",
        "user": "{source}",
    },
    {
        "system": "Nde ha'e peteĩ AI oikuaáva Guaraní porã. Embohova Guaraní-pe.",
        "user": "{source}",
    },
]

# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

TEMPLATES_SENTIMENT: list[dict[str, str]] = [
    {
        "system": "Clasificá el sentimiento del texto como: positivo, negativo o neutro.",
        "user": "Texto: {source}\nSentimiento:",
    },
    {
        "system": "Eres un analizador de sentimiento para textos en Guaraní y Español.",
        "user": "Determiná el sentimiento de este texto:\n{source}\n\nOpciones: positivo, negativo, neutro",
    },
    {
        "system": "Analizá el sentimiento del siguiente texto. Respondé solo con la etiqueta.",
        "user": "{source}\n\nSentimiento (positivo/negativo/neutro):",
    },
    {
        "system": "Emoha'anga mba'éichapa oñeñandu ko texto-pe.",
        "user": "Texto: {source}\nSentimiento (positivo/negativo/neutro):",
    },
    {
        "system": "Clasificador de sentimiento para Guaraní y Jopara.",
        "user": "¿Cuál es el sentimiento de este texto?\n\"{source}\"\nRespondé: positivo, negativo o neutro.",
    },
]

# ---------------------------------------------------------------------------
# Classification (humor, offense, emotion)
# ---------------------------------------------------------------------------

TEMPLATES_CLASSIFY_HUMOR: list[dict[str, str]] = [
    {
        "system": "Clasificá si el texto es humorístico o no.",
        "user": "Texto: {source}\n¿Es humor? (sí/no):",
    },
    {
        "system": "Detector de humor para textos en Guaraní y Español.",
        "user": "¿Este texto tiene intención humorística?\n\"{source}\"\nRespondé: sí o no.",
    },
]

TEMPLATES_CLASSIFY_OFFENSE: list[dict[str, str]] = [
    {
        "system": "Clasificá si el texto es ofensivo o no ofensivo.",
        "user": "Texto: {source}\n¿Es ofensivo? (sí/no):",
    },
    {
        "system": "Detector de discurso ofensivo para textos en Guaraní y Español.",
        "user": "Analizá si el siguiente texto es ofensivo:\n\"{source}\"\nRespondé: sí o no.",
    },
]

TEMPLATES_CLASSIFY_EMOTION: list[dict[str, str]] = [
    {
        "system": "Clasificá la emoción del texto: alegría, tristeza, enojo, miedo, sorpresa, neutro.",
        "user": "Texto: {source}\nEmoción:",
    },
    {
        "system": "Detector de emociones para textos en Guaraní y Español.",
        "user": "¿Qué emoción transmite este texto?\n\"{source}\"\nOpciones: alegría, tristeza, enojo, miedo, sorpresa, neutro.",
    },
]

# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

TEMPLATES_GENERATION: list[dict[str, str]] = [
    {
        "system": "Sos un escritor en Guaraní paraguayo. Escribí textos naturales y fluidos.",
        "user": "Escribí un texto en Guaraní sobre: {source}",
    },
    {
        "system": "Nde ha'e peteĩ escritor ohaíva Guaraní-pe. Ehai peteĩ texto.",
        "user": "Ehai Guaraní-pe ko tema rehegua: {source}",
    },
    {
        "system": "Generador de texto en Guaraní. Producí contenido natural y coherente.",
        "user": "Generá un párrafo en Guaraní sobre el siguiente tema:\n{source}",
    },
    {
        "system": "Nde ha'e peteĩ AI oikuaáva ohai Guaraní-pe.",
        "user": "Emombe'u chéve {source} rehegua Guaraní-pe.",
    },
    {
        "system": "Escritor creativo bilingüe. Escribí en Guaraní con naturalidad.",
        "user": "Tema: {source}\nEscribí un texto corto en Guaraní:",
    },
]


# ============================================================================
# Template registry & access functions
# ============================================================================

_TASK_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "translate_gn_es": TEMPLATES_TRANSLATE_GN_ES,
    "translate_es_gn": TEMPLATES_TRANSLATE_ES_GN,
    "chat": TEMPLATES_CHAT,
    "sentiment": TEMPLATES_SENTIMENT,
    "classify_humor": TEMPLATES_CLASSIFY_HUMOR,
    "classify_offense": TEMPLATES_CLASSIFY_OFFENSE,
    "classify_emotion": TEMPLATES_CLASSIFY_EMOTION,
    "generation": TEMPLATES_GENERATION,
}


def get_random_template(task: str) -> dict[str, str]:
    """Return a random prompt template for the given *task*.

    Parameters
    ----------
    task : str
        One of the registered task names:
        ``translate_gn_es``, ``translate_es_gn``, ``chat``,
        ``sentiment``, ``classify_humor``, ``classify_offense``,
        ``classify_emotion``, ``generation``.

    Returns
    -------
    dict[str, str]
        A dict with ``"system"`` and ``"user"`` keys.  The ``"user"``
        value contains a ``{source}`` placeholder for the input text.

    Raises
    ------
    ValueError
        If *task* is not registered.
    """
    templates = _TASK_TEMPLATES.get(task)
    if templates is None:
        valid = ", ".join(sorted(_TASK_TEMPLATES.keys()))
        raise ValueError(f"Unknown task {task!r}. Valid tasks: {valid}")
    return random.choice(templates)


def build_messages(
    task: str,
    source: str,
    *,
    template: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build a ChatML message list for a given task and input.

    Parameters
    ----------
    task : str
        Task name (see :func:`get_random_template`).
    source : str
        The input text to embed into the user prompt.
    template : dict[str, str] | None
        Specific template to use.  If ``None``, one is chosen at random.

    Returns
    -------
    list[dict[str, str]]
        Messages in ``[{"role": ..., "content": ...}, ...]`` format.
    """
    if template is None:
        template = get_random_template(task)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": template["user"].format(source=source)},
    ]
    return messages


def build_training_example(
    task: str,
    source: str,
    target: str,
    *,
    template: dict[str, str] | None = None,
) -> str:
    """Build a full ChatML training example including the assistant response.

    Parameters
    ----------
    task : str
        Task name.
    source : str
        Input text.
    target : str
        Expected assistant response.
    template : dict[str, str] | None
        Specific template.  If ``None``, chosen at random.

    Returns
    -------
    str
        Complete ChatML string with system, user, and assistant turns.
    """
    messages = build_messages(task, source, template=template)
    messages.append({"role": "assistant", "content": target})

    parts: list[str] = []
    for msg in messages:
        parts.append(CHATML_TEMPLATE.format(role=msg["role"], content=msg["content"]))
    return "".join(parts)


def list_tasks() -> list[str]:
    """Return a sorted list of all registered task names."""
    return sorted(_TASK_TEMPLATES.keys())

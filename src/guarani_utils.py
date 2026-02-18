"""Utility functions for Guarani NLP processing.

Provides language detection heuristics, token counting, fertility computation,
and character validation specific to Guarani (Avañe'e) text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# ---------------------------------------------------------------------------
# Guarani character set
# ---------------------------------------------------------------------------

# Core Latin letters used in Guarani orthography
_GUARANI_LETTERS = set("aãeẽiĩoõuũyỹAÃEẼIĨOÕUŨYỸ")

# Guarani-specific digraphs are checked at the string level, not single chars
# ch, mb, nd, ng, nt, nk, rr, etc.

# The tilde g (g̃ / G̃) is a combining sequence: g + U+0303
# We also accept the rare pre-composed variants if any font produces them.
_COMBINING_TILDE = "\u0303"

# Full set of characters that may appear in valid Guarani text
GUARANI_CHARS: set[str] = (
    _GUARANI_LETTERS
    | set("bcdfghjklmnñpqrstvwxyzBCDFGHJKLMNÑPQRSTVWXYZ")
    | {_COMBINING_TILDE}
    | set("'- ")  # apostrophe (puso), hyphen, space
    | set("0123456789")
    | set(".,;:!?()\"")
)

# ---------------------------------------------------------------------------
# Guarani-specific markers used for language detection
# ---------------------------------------------------------------------------

# Nasal vowels that rarely appear in Spanish
_NASAL_VOWEL_PATTERN = re.compile(r"[ãẽĩõũỹÃẼĨÕŨỸ]|[gG]\u0303")

# Puso (glottal stop) represented by apostrophe in Guarani words
_PUSO_PATTERN = re.compile(r"[a-záéíóúãẽĩõũỹ]'[a-záéíóúãẽĩõũỹ]", re.IGNORECASE)

# High-frequency Guarani function words
_GUARANI_WORDS: set[str] = {
    "ha", "ko", "kova", "pe", "gui", "rupi", "rehe", "ndive",
    "oĩ", "heta", "michĩ", "porã", "vai", "guasu", "mitã",
    "kuña", "kuimba'e", "ñande", "ore", "peẽ", "ha'e",
    "che", "nde", "upe", "ko'ã", "upéi", "avei",
    "oiko", "oĩ", "oguahẽ", "omombe'u", "oñepyrũ",
    "jey", "katu", "voi", "niko", "ndaje",
    "mba'e", "mba'éichapa", "mba'érepa", "moõpa",
    "ára", "oga", "tetã", "yvy", "ysyry",
    "guarani", "guaraní", "avañe'ẽ",
    "opavave", "opaite", "hína", "hikuái",
    "rehegua", "ndaha'éi", "ikatu",
}

# High-frequency Spanish words (to detect Spanish)
_SPANISH_WORDS: set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "con", "por", "para", "que", "es",
    "son", "está", "están", "fue", "ser", "haber", "tiene",
    "como", "pero", "más", "este", "esta", "estos", "estas",
    "también", "muy", "ya", "hay", "desde", "hasta", "entre",
}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect whether *text* is Guarani, Spanish, or Jopara (mixed).

    Uses a simple heuristic based on:
    - Presence of nasal vowels and puso (glottal stop)
    - Frequency of language-specific function words

    Returns
    -------
    str
        One of ``"guarani"``, ``"spanish"``, or ``"jopara"``.
    """
    if not text or not text.strip():
        return "spanish"

    text_lower = text.lower()
    words = set(re.findall(r"[\w'ãẽĩõũỹ]+", text_lower, re.UNICODE))

    # Count Guarani markers
    nasal_count = len(_NASAL_VOWEL_PATTERN.findall(text))
    puso_count = len(_PUSO_PATTERN.findall(text))
    gn_word_count = len(words & _GUARANI_WORDS)
    gn_score = nasal_count * 2 + puso_count * 2 + gn_word_count

    # Count Spanish markers
    es_word_count = len(words & _SPANISH_WORDS)
    es_score = es_word_count

    total = gn_score + es_score
    if total == 0:
        return "spanish"

    gn_ratio = gn_score / total

    if gn_ratio > 0.7:
        return "guarani"
    elif gn_ratio < 0.3:
        return "spanish"
    else:
        return "jopara"


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer: Any) -> int:
    """Count the number of tokens produced by *tokenizer* for *text*.

    Parameters
    ----------
    text : str
        Input text.
    tokenizer : Any
        A HuggingFace-compatible tokenizer with an ``encode`` method.

    Returns
    -------
    int
        Number of tokens.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_fertility(text: str, tokenizer: Any) -> float:
    """Compute the token fertility (tokens per word) for *text*.

    A lower fertility means the tokenizer is more efficient for the
    language.  Guarani typically has higher fertility than English or
    Spanish on standard tokenizers.

    Parameters
    ----------
    text : str
        Input text.
    tokenizer : Any
        A HuggingFace-compatible tokenizer.

    Returns
    -------
    float
        Tokens-per-word ratio.  Returns ``0.0`` if *text* is empty.
    """
    words = text.split()
    if not words:
        return 0.0
    n_tokens = count_tokens(text, tokenizer)
    return n_tokens / len(words)


# ---------------------------------------------------------------------------
# Guarani text validation
# ---------------------------------------------------------------------------

def is_valid_guarani(text: str) -> bool:
    """Check whether *text* contains only valid Guarani characters.

    Allows standard punctuation, digits, whitespace, and the full
    Guarani alphabet including nasal vowels (a\u0303, e\u0303, etc.)
    and the puso (apostrophe as glottal stop).

    Parameters
    ----------
    text : str
        Text to validate.

    Returns
    -------
    bool
        ``True`` if every character in *text* is valid for Guarani writing.
    """
    if not text:
        return False

    for char in unicodedata.normalize("NFC", text):
        if char in GUARANI_CHARS:
            continue
        # Allow common whitespace
        if char in {"\n", "\r", "\t"}:
            continue
        # Allow Unicode combining tilde (U+0303) on any letter
        if unicodedata.category(char).startswith("M"):
            continue
        # Allow accented vowels common in loanwords
        if char in set("áéíóúÁÉÍÓÚ"):
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_guarani(text: str) -> str:
    """Normalize Guarani text to NFC form and standardize apostrophes.

    - Converts all Unicode to NFC.
    - Replaces curly/typographic apostrophes with the standard ASCII
      apostrophe (U+0027) used for puso.
    - Collapses multiple whitespace into single spaces.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Normalized text.
    """
    text = unicodedata.normalize("NFC", text)
    # Standardize apostrophe variants
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u0060", "'").replace("\u00B4", "'")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

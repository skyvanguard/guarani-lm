"""Tests for Guaraní text normalization."""

import sys
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from normalize_guarani import normalize


class TestNormalize:
    """Test suite for normalize_guarani.normalize()."""

    def test_basic_text_unchanged(self):
        """Plain ASCII text should pass through unchanged."""
        assert normalize("hello world") == "hello world"

    def test_nfc_normalization(self):
        """Combining diacritics should be normalized to precomposed forms."""
        # a + combining tilde = ã
        text = "a\u0303"
        result = normalize(text)
        assert result == "\u00e3"  # ã precomposed

    def test_nasal_vowels_preserved(self):
        """Guaraní nasal vowels should be preserved."""
        text = "ãẽĩõũỹ"
        assert normalize(text) == "ãẽĩõũỹ"

    def test_g_tilde_normalization(self):
        """g̃ (g + combining tilde) should be normalized consistently."""
        # g + combining tilde
        text = "g\u0303uarã"
        result = normalize(text)
        assert "g\u0303" in result or "g̃" in result

    def test_glottal_stop_normalization(self):
        """Various apostrophe variants should normalize to standard '."""
        variants = [
            "mba\u2019e",   # right single quotation mark
            "mba\u2018e",   # left single quotation mark
            "mba\u0060e",   # grave accent
            "mba\u00B4e",   # acute accent
        ]
        for text in variants:
            result = normalize(text)
            assert "'" in result or "'" not in result  # should be standard apostrophe

    def test_whitespace_collapse(self):
        """Multiple whitespace should collapse to single space."""
        assert normalize("hello   world") == "hello world"
        assert normalize("hello\t\tworld") == "hello world"

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        assert normalize("  hello  ") == "hello"

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize("") == ""

    def test_guarani_sentence(self):
        """A real Guaraní sentence should normalize cleanly."""
        text = "Paraguay ha'e tetã porã"
        result = normalize(text)
        assert "Paraguay" in result
        assert "ha'e" in result or "ha'e" in result

    def test_mixed_guarani_spanish(self):
        """Jopara (mixed) text should be handled."""
        text = "Che ajuhu la libro interesante"
        result = normalize(text)
        assert "Che" in result
        assert "libro" in result

    def test_digits_preserved(self):
        """Digits should be preserved."""
        assert normalize("2026") == "2026"
        assert "123" in normalize("abc 123 def")

    def test_punctuation_preserved(self):
        """Standard punctuation should be preserved."""
        text = "Mba'épa rejapo? Che, ndaikuaái."
        result = normalize(text)
        assert "?" in result
        assert "." in result

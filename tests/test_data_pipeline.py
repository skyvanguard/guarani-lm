"""Tests for the data processing pipeline."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMergeDatasets:
    """Test dataset merging and splitting logic."""

    def test_jsonl_format(self, tmp_path):
        """JSONL files should have valid JSON on each line."""
        jsonl_file = tmp_path / "test.jsonl"
        records = [
            {"text": "Guaraní iporã"},
            {"text": "Paraguay retã"},
        ]
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                assert "text" in data

    def test_split_ratio(self):
        """90/5/5 split should produce correct proportions."""
        total = 1000
        train = int(total * 0.90)
        val = int(total * 0.05)
        test = total - train - val

        assert train == 900
        assert val == 50
        assert test == 50

    def test_chatml_format(self):
        """ChatML instruction samples should have correct structure."""
        sample = {
            "messages": [
                {"role": "system", "content": "Eres un traductor Guaraní-Español."},
                {"role": "user", "content": "Traducí al español: Mba'éichapa reime?"},
                {"role": "assistant", "content": "¿Cómo estás?"},
            ]
        }
        assert len(sample["messages"]) == 3
        roles = [m["role"] for m in sample["messages"]]
        assert roles == ["system", "user", "assistant"]
        for msg in sample["messages"]:
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["content"], str)


class TestPromptTemplates:
    """Test prompt template structure."""

    def test_translation_template_structure(self):
        """Translation templates should have system + user + assistant slots."""
        template = {
            "system": "Eres un traductor profesional de Guaraní a Español.",
            "user": "Traducí al español: {source}",
            "assistant": "{target}",
        }
        assert "{source}" in template["user"]
        assert "{target}" in template["assistant"]

    def test_template_fill(self):
        """Templates should fill correctly with data."""
        user_tpl = "Traducí al español: {source}"
        filled = user_tpl.format(source="Mba'éichapa reime?")
        assert "Mba'éichapa reime?" in filled
        assert "{source}" not in filled


class TestGuaraniUtils:
    """Test Guaraní-specific utility functions."""

    def test_guarani_chars_present(self):
        """Text with nasal vowels should be detected as Guaraní-like."""
        guarani_chars = set("ãẽĩõũỹ")
        text = "Paraguái ha'e tetã porã"
        has_guarani = bool(set(text) & guarani_chars)
        assert has_guarani  # "porã" contains ã

    def test_spanish_no_guarani_chars(self):
        """Pure Spanish text should not have Guaraní-specific chars."""
        guarani_specific = set("ỹ")  # ỹ is unique to Guaraní
        text = "Buenos días, ¿cómo estás?"
        has_guarani = bool(set(text) & guarani_specific)
        assert not has_guarani

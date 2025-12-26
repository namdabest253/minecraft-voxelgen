"""Tests for BlockVocabulary."""

import tempfile
from pathlib import Path

import pytest

from src.data.vocabulary import BlockVocabulary


class TestBlockVocabulary:
    """Test cases for BlockVocabulary class."""

    def test_init_has_air(self) -> None:
        """Vocabulary should start with air as ID 0."""
        vocab = BlockVocabulary()
        assert vocab.get_id("minecraft:air") == 0
        assert vocab.get_block(0) == "minecraft:air"

    def test_add_block(self) -> None:
        """Adding blocks should assign sequential IDs."""
        vocab = BlockVocabulary()
        id1 = vocab.add_block("minecraft:stone")
        id2 = vocab.add_block("minecraft:dirt")

        assert id1 == 1
        assert id2 == 2

    def test_add_block_idempotent(self) -> None:
        """Adding same block twice should return same ID."""
        vocab = BlockVocabulary()
        id1 = vocab.add_block("minecraft:stone")
        id2 = vocab.add_block("minecraft:stone")

        assert id1 == id2 == 1

    def test_get_id(self) -> None:
        """get_id should return correct ID."""
        vocab = BlockVocabulary()
        vocab.add_block("minecraft:stone")

        assert vocab.get_id("minecraft:stone") == 1

    def test_get_id_unknown_raises(self) -> None:
        """get_id should raise KeyError for unknown blocks."""
        vocab = BlockVocabulary()

        with pytest.raises(KeyError):
            vocab.get_id("minecraft:unknown")

    def test_get_block(self) -> None:
        """get_block should return correct name."""
        vocab = BlockVocabulary()
        vocab.add_block("minecraft:stone")

        assert vocab.get_block(1) == "minecraft:stone"

    def test_get_block_unknown_raises(self) -> None:
        """get_block should raise KeyError for unknown IDs."""
        vocab = BlockVocabulary()

        with pytest.raises(KeyError):
            vocab.get_block(999)

    def test_size(self) -> None:
        """size should return correct count."""
        vocab = BlockVocabulary()
        assert vocab.size == 1  # Just air

        vocab.add_block("minecraft:stone")
        vocab.add_block("minecraft:dirt")
        assert vocab.size == 3

    def test_save_and_load(self) -> None:
        """Vocabulary should round-trip through save/load."""
        vocab = BlockVocabulary()
        vocab.add_block("minecraft:stone")
        vocab.add_block("minecraft:dirt")
        vocab.add_block("minecraft:oak_planks")

        with tempfile.TemporaryDirectory() as tmpdir:
            vocab.save(tmpdir)
            loaded = BlockVocabulary.load(tmpdir)

        assert loaded.size == vocab.size
        assert loaded.get_id("minecraft:stone") == 1
        assert loaded.get_id("minecraft:dirt") == 2
        assert loaded.get_id("minecraft:oak_planks") == 3
        assert loaded.get_block(0) == "minecraft:air"

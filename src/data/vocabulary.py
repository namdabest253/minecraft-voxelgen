"""Block vocabulary management."""

import json
from pathlib import Path


class BlockVocabulary:
    """Manages mapping between block names and numeric IDs.

    The vocabulary assigns a unique integer ID to each Minecraft block type.
    Air is always ID 0.

    Attributes:
        block_to_id: Dict mapping block names to IDs
        id_to_block: Dict mapping IDs to block names
        size: Number of unique blocks in vocabulary

    Example:
        >>> vocab = BlockVocabulary()
        >>> vocab.add_block("minecraft:stone")
        >>> vocab.add_block("minecraft:dirt")
        >>> vocab.get_id("minecraft:stone")
        1
        >>> vocab.get_block(1)
        'minecraft:stone'
    """

    def __init__(self) -> None:
        """Initialize an empty vocabulary with air as ID 0."""
        self.block_to_id: dict[str, int] = {"minecraft:air": 0}
        self.id_to_block: dict[int, str] = {0: "minecraft:air"}
        self._next_id = 1

    @property
    def size(self) -> int:
        """Return the number of unique blocks in the vocabulary."""
        return len(self.block_to_id)

    def add_block(self, block_name: str) -> int:
        """Add a block to the vocabulary if not already present.

        Args:
            block_name: Full block name (e.g., "minecraft:stone")

        Returns:
            The ID assigned to this block
        """
        if block_name not in self.block_to_id:
            self.block_to_id[block_name] = self._next_id
            self.id_to_block[self._next_id] = block_name
            self._next_id += 1
        return self.block_to_id[block_name]

    def get_id(self, block_name: str) -> int:
        """Get the ID for a block name.

        Args:
            block_name: Full block name

        Returns:
            The numeric ID for this block

        Raises:
            KeyError: If the block is not in the vocabulary
        """
        return self.block_to_id[block_name]

    def get_block(self, block_id: int) -> str:
        """Get the block name for an ID.

        Args:
            block_id: Numeric block ID

        Returns:
            The full block name

        Raises:
            KeyError: If the ID is not in the vocabulary
        """
        return self.id_to_block[block_id]

    def save(self, directory: str | Path) -> None:
        """Save vocabulary to JSON files.

        Args:
            directory: Directory to save files to
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "block_to_id.json", "w") as f:
            json.dump(self.block_to_id, f, indent=2)

        with open(directory / "id_to_block.json", "w") as f:
            # JSON keys must be strings
            json.dump({str(k): v for k, v in self.id_to_block.items()}, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "BlockVocabulary":
        """Load vocabulary from JSON files.

        Args:
            directory: Directory containing vocabulary files

        Returns:
            Loaded BlockVocabulary instance
        """
        directory = Path(directory)

        vocab = cls()

        with open(directory / "block_to_id.json") as f:
            vocab.block_to_id = json.load(f)

        with open(directory / "id_to_block.json") as f:
            vocab.id_to_block = {int(k): v for k, v in json.load(f).items()}

        vocab._next_id = max(vocab.id_to_block.keys()) + 1

        return vocab

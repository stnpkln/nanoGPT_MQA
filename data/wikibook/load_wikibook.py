
from pathlib import Path

import numpy as np
from directories import BOOKCORPUS_DIR, WIKITEXT_DIR


def load_wikibook(
    split: str = "train",
    wikitext_tokens: int = 1_000_000,
    bookcorpus_tokens: int = 1_000_000,
    seed: int = 31,
    shuffle: bool = True,
) -> np.ndarray:
    """
    Load a mixed WikiText-2 + BookCorpus token dataset.

    Args:
        split: "train", "val", or "test".
        wikitext_tokens: Number of tokens to take from WikiText-2.
        bookcorpus_tokens: Number of tokens to take from BookCorpus.
        seed: Random seed for reproducibility.
        shuffle: Whether to shuffle the combined token stream.

    Returns:
        A uint16 NumPy array containing the requested token counts.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")

    if wikitext_tokens < 0 or bookcorpus_tokens < 0:
        raise ValueError("Token counts cannot be negative")

    rng = np.random.default_rng(seed)

    def load_tokens(path: Path, count: int) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        tokens = np.memmap(path, dtype=np.uint16, mode="r")

        if count > len(tokens):
            raise ValueError(
                f"{path} contains {len(tokens):,} tokens, "
                f"but {count:,} were requested."
            )

        if count == 0:
            return np.empty(0, dtype=np.uint16)

        # Select a contiguous region at a random offset.
        start = int(rng.integers(0, len(tokens) - count + 1))
        return np.array(tokens[start : start + count], copy=True)

    wiki = load_tokens(WIKITEXT_DIR / f"{split}.bin", wikitext_tokens)
    books = load_tokens(BOOKCORPUS_DIR / f"{split}.bin", bookcorpus_tokens)

    mixed = np.concatenate((wiki, books))

    if shuffle:
        rng.shuffle(mixed)

    return mixed

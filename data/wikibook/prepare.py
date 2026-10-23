
import argparse
from pathlib import Path

import numpy as np
from directories import BOOKCORPUS_DIR, WIKIBOOK_DIR, WIKITEXT_DIR
from load_wikibook import load_wikibook


def get_token_count(path: Path) -> int:
    """Return the number of tokens in a uint16 binary file."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return path.stat().st_size // np.dtype(np.uint16).itemsize


def create_wikibook(
    wiki_percentage: float,
    book_percentage: float,
    seed: int = 42,
):
    if not 0 <= wiki_percentage <= 100:
        raise ValueError("WikiText-2 percentage must be between 0 and 100")

    if not 0 <= book_percentage <= 100:
        raise ValueError("BookCorpus percentage must be between 0 and 100")

    wiki_total = get_token_count(WIKITEXT_DIR / "train.bin")
    book_total = get_token_count(BOOKCORPUS_DIR / "train.bin")

    wiki_tokens = int(wiki_total * wiki_percentage / 100)
    book_tokens = int(book_total * book_percentage / 100)

    dataset_name = (
        f"wiki_{wiki_percentage:g}_book_{book_percentage:g}_train.bin"
    )
    output_file = WIKIBOOK_DIR / dataset_name

    print(f"WikiText-2: {wiki_tokens:,} / {wiki_total:,} tokens")
    print(f"BookCorpus: {book_tokens:,} / {book_total:,} tokens")

    tokens = load_wikibook(
        split="train",
        wikitext_tokens=wiki_tokens,
        bookcorpus_tokens=book_tokens,
        seed=seed,
        shuffle=True,
    )

    tokens.astype(np.uint16, copy=False).tofile(output_file)

    print(f"Saved {output_file.name}: {len(tokens):,} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a mixed WikiText-2 and BookCorpus training dataset."
    )
    parser.add_argument("--wiki", type=float, default=20)
    parser.add_argument("--book", type=float, default=80)
    parser.add_argument("--seed", type=int, default=31)

    args = parser.parse_args()

    create_wikibook(args.wiki, args.book, args.seed)

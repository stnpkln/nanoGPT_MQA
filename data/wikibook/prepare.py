
import argparse
from pathlib import Path

import numpy as np
from directories import BOOKCORPUS_DIR, WIKIBOOK_DIR, WIKITEXT_DIR
from load_wikibook import load_wikibook


def get_token_count(path: Path) -> int:
    """Return the number of tokens in a uint16 binary file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    itemsize = np.dtype(np.uint16).itemsize
    if path.stat().st_size % itemsize != 0:
        raise ValueError(f"Invalid uint16 binary file size: {path}")

    return path.stat().st_size // itemsize


def create_wikibook(
    wiki_percentage: float,
    book_percentage: float,
    seed: int = 31,
):
    for name, percentage in (
        ("WikiText-2", wiki_percentage),
        ("BookCorpus", book_percentage),
    ):
        if not 0 <= percentage <= 100:
            raise ValueError(f"{name} percentage must be between 0 and 100")

    results = {}

    for split in ("train", "val"):
        wiki_total = get_token_count(WIKITEXT_DIR / f"{split}.bin")
        book_total = get_token_count(BOOKCORPUS_DIR / f"{split}.bin")

        wiki_tokens = int(wiki_total * wiki_percentage / 100)
        book_tokens = int(book_total * book_percentage / 100)

        print(f"\nCreating {split} split")
        print(
            f"WikiText-2: {wiki_tokens:,} / {wiki_total:,} tokens "
            f"({wiki_percentage:g}%)"
        )
        print(
            f"BookCorpus: {book_tokens:,} / {book_total:,} tokens "
            f"({book_percentage:g}%)"
        )

        tokens = load_wikibook(
            split=split,
            wikitext_tokens=wiki_tokens,
            bookcorpus_tokens=book_tokens,
            seed=seed,
            shuffle=True,
        )

        output_file = WIKIBOOK_DIR / f"{split}.bin"
        tokens.astype(np.uint16, copy=False).tofile(output_file)

        results[split] = {
            "wiki_total": wiki_total,
            "book_total": book_total,
            "wiki_used": wiki_tokens,
            "book_used": book_tokens,
            "total_used": len(tokens),
        }

        print(f"Saved {output_file}: {len(tokens):,} tokens")

    info_file = WIKIBOOK_DIR / "dataset_info.txt"

    with info_file.open("w", encoding="utf-8") as f:
        f.write("WikiBook dataset configuration\n")
        f.write("==============================\n\n")
        f.write(f"WikiText-2 percentage: {wiki_percentage:g}%\n")
        f.write(f"BookCorpus percentage: {book_percentage:g}%\n")
        f.write(f"Random seed: {seed}\n")
        f.write("Shuffle: True\n")
        f.write("Token dtype: uint16\n")
        f.write("Splits: train, val\n\n")

        for split, data in results.items():
            f.write(f"{split.upper()} SPLIT\n")
            f.write(f"  WikiText-2 available: {data['wiki_total']:,}\n")
            f.write(f"  BookCorpus available: {data['book_total']:,}\n")
            f.write(f"  WikiText-2 used: {data['wiki_used']:,}\n")
            f.write(f"  BookCorpus used: {data['book_used']:,}\n")
            f.write(f"  Total tokens: {data['total_used']:,}\n\n")

    print(f"\nDataset metadata saved to {info_file}")
    print("WikiBook dataset creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create mixed WikiText-2 and BookCorpus train/validation data."
    )
    parser.add_argument("--wiki", type=float, default=20)
    parser.add_argument("--book", type=float, default=80)
    parser.add_argument("--seed", type=int, default=31)

    args = parser.parse_args()
    create_wikibook(args.wiki, args.book, args.seed)

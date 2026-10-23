from pathlib import Path

WIKIBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = WIKIBOOK_DIR.parent
WIKITEXT_DIR = DATA_DIR / "wikitext2"
BOOKCORPUS_DIR = DATA_DIR / "bookcorpus"
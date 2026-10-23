
import os

import numpy as np
import tiktoken
from datasets import load_dataset

# Directory containing this script and the output files
data_dir = os.path.dirname(os.path.abspath(__file__))

# Load WikiText-2 with the original text preserved
dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
)

# GPT-2 byte-pair encoding tokenizer
enc = tiktoken.get_encoding("gpt2")


def encode_split(split):
    # Each example contains a "text" field.
    # Join lines to preserve document boundaries and formatting.
    text = "\n".join(dataset[split]["text"])

    # Encode using GPT-2 BPE
    token_ids = enc.encode_ordinary(text)

    print(f"{split} has {len(token_ids):,} tokens")

    # GPT-2 token IDs fit in uint16
    return np.array(token_ids, dtype=np.uint16)


# Encode the official dataset splits independently
train_ids = encode_split("train")
val_ids = encode_split("validation")
test_ids = encode_split("test")

# Export to binary files
train_ids.tofile(os.path.join(data_dir, "wikitext2_train.bin"))
val_ids.tofile(os.path.join(data_dir, "wikitext2_val.bin"))
test_ids.tofile(os.path.join(data_dir, "wikitext2_test.bin"))

print("Saved train.bin, val.bin, and test.bin")


import os

import numpy as np
import tiktoken
from datasets import load_dataset

dataset_folder = os.path.dirname(os.path.abspath(__file__))

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
)

encoding = tiktoken.get_encoding("gpt2")

def encode(split):
    # no need to use batches, its a small dataset
    text = "\n".join(dataset[split]["text"])
    token_ids = encoding.encode_ordinary(text)

    return np.array(token_ids, dtype=np.uint16)

train_ids = encode("train")
val_ids = encode("validation")
test_ids = encode("test")

train_ids.tofile(os.path.join(dataset_folder, "wikitext2_train.bin"))
val_ids.tofile(os.path.join(dataset_folder, "wikitext2_val.bin"))
test_ids.tofile(os.path.join(dataset_folder, "wikitext2_test.bin"))

print("Wikitext2 encoding done and saved.")

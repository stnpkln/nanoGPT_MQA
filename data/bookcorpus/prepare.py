
import os

import numpy as np
import tiktoken
from datasets import load_dataset

dataset_folder = os.path.dirname(os.path.abspath(__file__))
encoding = tiktoken.get_encoding("gpt2")

train = load_dataset(
    "SamuelYang/bookcorpus",
    split="train[:90%]",
)

val = load_dataset(
    "SamuelYang/bookcorpus",
    split="train[90%:95%]",
)

test = load_dataset(
    "SamuelYang/bookcorpus",
    split="train[95%:]",
)


def encode(dataset, output_file, batch_size=1000):
    total_tokens = 0

    with open(output_file, "wb") as f:
        # tokenization in batches to not overwhelm the memory. Its a pretty big dataset.
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]

            texts = [text for text in batch["text"] if text]
            text = "\n".join(texts)

            if not text:
                continue

            token_ids = encoding.encode_ordinary(text)
            token_np_array = np.asarray(token_ids, dtype=np.uint16)

            token_np_array.tofile(f)
            total_tokens += len(token_np_array)

    print(f"Saved {output_file}: {total_tokens} tokens")


encode(train, os.path.join(dataset_folder, "bookcorpus_train.bin"))
encode(val, os.path.join(dataset_folder, "bookcorpus_val.bin"))
encode(test, os.path.join(dataset_folder, "bookcorpus_test.bin"))

print("Bookcorpus encoding done and saved.")

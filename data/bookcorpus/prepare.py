
import os

import numpy as np
import tiktoken
from datasets import load_dataset

data_dir = os.path.dirname(os.path.abspath(__file__))
enc = tiktoken.get_encoding("gpt2")

# Load BookCorpus and create 90/5/5 splits.
train_data = load_dataset(
    "SamuelYang/bookcorpus",
    split="train[:90%]",
)

val_data = load_dataset(
    "SamuelYang/bookcorpus",
    split="train[90%:95%]",
)

test_data = load_dataset(
    "SamuelYang/bookcorpus",
    split="train[95%:]",
)


def encode_split(dataset, output_file, batch_size=1000):
    total_tokens = 0

    with open(output_file, "wb") as f:
        # Do the tokenization in batches to not overwhelm the memory. Its a pretty big dataset.
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]

            texts = [text for text in batch["text"] if text]
            text = "\n".join(texts)

            if not text:
                continue

            token_ids = enc.encode_ordinary(text)
            token_array = np.asarray(token_ids, dtype=np.uint16)

            token_array.tofile(f)
            total_tokens += len(token_array)

            if start % (batch_size * 10) == 0:
                print(
                    f"{os.path.basename(output_file)}: "
                    f"{total_tokens:,} tokens processed"
                )

    print(f"Saved {output_file}: {total_tokens:,} tokens")


encode_split(train_data, os.path.join(data_dir, "bookcorpus_train.bin"))
encode_split(val_data, os.path.join(data_dir, "bookcorpus_val.bin"))
encode_split(test_data, os.path.join(data_dir, "bookcorpus_test.bin"))

print("BookCorpus preprocessing complete.")

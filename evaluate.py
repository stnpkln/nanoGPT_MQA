
import os
import argparse
import json
import time
import math

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


# constants for things that are unlikely to change from run to run
WARMUP_RUNS = 10
REPEATS = 1000
DTYPE = torch.float16
DEVICE = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--kv_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--gen_tokens", type=int, default=100)
    return parser.parse_args()


def estimate_kv_cache_size(config, sequence_length):
    head_dimension = config.n_embd // config.n_head
    n_kv_heads = 1 if config.mqa else config.n_head
    bytes_per_element = torch.tensor([], dtype=DTYPE).element_size()
    size = (2 * config.n_layer * n_kv_heads * sequence_length * head_dimension * bytes_per_element)

    return size


@torch.no_grad()
def measure_inference_time_and_memory(model, data, prompt_length, gen_tokens):
    if (prompt_length + gen_tokens) > model.config.block_size:
        raise ValueError("Prompt length plus generated tokens exceeds model block size.")
    if len(data) < prompt_length:
        raise ValueError("Prompt length exceeds available dataset tokens.")

    prompt = torch.as_tensor(
        np.asarray(data[:prompt_length], dtype=np.int64),
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    def generate():
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            return model.generate(prompt, max_new_tokens=gen_tokens)

    for _ in range(WARMUP_RUNS):
        output = generate()
        del output

    torch.cuda.synchronize()

    inference_times = []
    peak_allocated = 0

    for _ in range(REPEATS):
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        output = generate()
        torch.cuda.synchronize()

        inference_times.append(time.perf_counter() - start)
        del output

        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())

    average_seconds = sum(inference_times) / REPEATS
    peak_increment = max(0, peak_allocated - baseline_allocated)

    return {
        "prompt_length": prompt_length,
        "generated_tokens": gen_tokens,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": REPEATS,
        "ms_per_token_average": average_seconds * 1000 / gen_tokens,
        "kv_cache_enabled": bool(model.config.kv_cache),
        "estimated_kv_cache_mib": estimate_kv_cache_size(model.config, prompt_length + gen_tokens) / (1024 ** 2),
        "gpu_peak_allocated_mib": peak_allocated / (1024 ** 2),
        "gpu_peak_increment_over_baseline_mib": peak_increment / (1024 ** 2),
    }


def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    
    args = parse_args()

    checkpoint_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = dict(checkpoint["model_args"])
    model_args["kv_cache"] = args.kv_cache

    config = GPTConfig(**model_args)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_path = os.path.join("data", args.dataset, f"{args.split}.bin")
    evaluation_data = np.memmap(data_path, dtype=np.uint16, mode="r")

    block_size = config.block_size
    num_chunks = (len(evaluation_data) - 1) // block_size

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for first in range(0, num_chunks, args.batch_size):
            last = min(first + args.batch_size, num_chunks)
            sequences= []
            targets = []

            for chunk_idx in range(first, last):
                start = chunk_idx * block_size
                tokens = torch.from_numpy(np.array(evaluation_data[start:(start + block_size + 1)], dtype=np.int64))
                sequences.append(tokens[:-1])
                targets.append(tokens[1:])

            x = torch.stack(sequences).to(DEVICE)
            y = torch.stack(targets).to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=DTYPE):
                logits, _, _ = model(x, y)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    average_loss = total_loss / total_tokens

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "attention": "MQA" if config.mqa else "MHA",
        "kv_cache_enabled": args.kv_cache,
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_evaluated": total_tokens,
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
        "inference": measure_inference_time_and_memory(model, evaluation_data, args.prompt_length, args.gen_tokens),
    }

    if not args.save_json:
        print("Done.")

    output_path = os.path.join(args.out_dir, f"evaluation_{'kv' if args.kv_cache else 'no_kv'}.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4)

if __name__ == "__main__":
    evaluate()

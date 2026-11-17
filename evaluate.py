
import argparse
import json
import math
import os

import numpy as np
import torch

from model import GPT, GPTConfig

import time
import torch


@torch.no_grad()
def benchmark_inference(
    model,
    data,
    device,
    dtype,
    prompt_length=128,
    gen_tokens=100,
    warmup=5,
    repeats=20,
):
    model.eval()

    # Use batch size 1 for autoregressive generation
    prompt = torch.tensor(
        data[:prompt_length],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    if prompt_length + gen_tokens > model.config.block_size:
        raise ValueError(
            "prompt_length + gen_tokens must not exceed block_size."
        )

    use_amp = device.type == "cuda" and dtype != torch.float32

    def generate():
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=use_amp,
        ):
            return model.generate(
                prompt,
                max_new_tokens=gen_tokens,
            )

    # Warmup runs (not included in timing)
    for _ in range(warmup):
        generate()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure generation time
    times = []

    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        generate()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    ms_per_token = (avg_time / gen_tokens) * 1000
    tokens_per_second = gen_tokens / avg_time

    print("\n--- Inference benchmark ---")
    print(f"Prompt length:       {prompt_length} tokens")
    print(f"Generated tokens:    {gen_tokens}")
    print(f"Number of runs:      {repeats}")
    print(f"Average total time:  {avg_time:.4f} seconds")
    print(f"Time per token:      {ms_per_token:.3f} ms")
    print(f"Generation speed:    {tokens_per_second:.2f} tokens/sec")

    return {
        "avg_time_seconds": avg_time,
        "ms_per_token": ms_per_token,
        "tokens_per_second": tokens_per_second,
    }


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--benchmark_inference", action="store_true")
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--gen_tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    # Device and precision
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda" and args.dtype in ("float16", "bfloat16")
    amp_dtype = (
        torch.float16 if args.dtype == "float16"
        else torch.bfloat16
    )

    # Load checkpoint
    checkpoint_path = os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_args = checkpoint["model_args"]

    # Ensure compatibility with model configurations
    model_args.setdefault("mqa", False)

    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Attention: {'MQA' if config.mqa else 'MHA'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenized test data
    data_path = os.path.join("data", args.dataset, f"{args.split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    block_size = config.block_size

    # Number of complete chunks; each chunk predicts block_size tokens
    num_chunks = (len(data) - 1) // block_size

    if num_chunks == 0:
        raise ValueError("Test dataset is too small for the model's block_size.")

    total_loss = 0.0
    total_tokens = 0

    print(f"Evaluating {num_chunks * block_size:,} tokens...")

    with torch.no_grad():
        for first_chunk in range(0, num_chunks, args.batch_size):
            last_chunk = min(first_chunk + args.batch_size, num_chunks)

            sequences = []
            targets = []

            for chunk_idx in range(first_chunk, last_chunk):
                start = chunk_idx * block_size
                end = start + block_size + 1

                tokens = torch.from_numpy(
                    np.array(data[start:end], dtype=np.int64)
                )

                sequences.append(tokens[:-1])
                targets.append(tokens[1:])

            x = torch.stack(sequences).to(device)
            y = torch.stack(targets).to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp
            ):
                logits, _ = model(x, y)

            # Calculate summed cross-entropy, not just batch mean
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    if args.benchmark_inference:
        benchmark_inference(
            model=model,
            data=data,
            device=device,
            dtype=amp_dtype,
            prompt_length=args.prompt_length,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "attention": "MQA" if config.mqa else "MHA",
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_evaluated": total_tokens,
        "loss": average_loss,
        "perplexity": perplexity,
    }

    print("\n===== Evaluation results =====")
    print(f"Dataset:          {results['dataset']}")
    print(f"Split:            {results['split']}")
    print(f"Attention:        {results['attention']}")
    print(f"Parameters:       {results['parameters']:,}")
    print(f"Tokens evaluated: {results['tokens_evaluated']:,}")
    print(f"Test loss:        {results['loss']:.4f}")
    print(f"Perplexity:       {results['perplexity']:.4f}")

    if args.save_json:
        os.makedirs(args.out_dir, exist_ok=True)
        output_path = os.path.join(args.out_dir, "evaluation_results.json")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    evaluate()

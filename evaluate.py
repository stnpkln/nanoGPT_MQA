import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a nanoGPT checkpoint and optionally benchmark generation."
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--save_json", action="store_true")

    parser.add_argument("--benchmark_inference", action="store_true")
    # Supports both --kv_cache and --no-kv_cache.
    parser.add_argument(
        "--kv_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable KV caching for generation (default: disabled).",
    )
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--gen_tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def get_autocast_settings(device, dtype_name):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    amp_dtype = dtype_map[dtype_name]
    use_amp = device.type == "cuda" and dtype_name != "float32"
    return amp_dtype, use_amp


def theoretical_kv_cache_size(model, batch_size, sequence_length, cache_dtype):
    """
    Estimate the KV tensor size, assuming the cache stores:
    [batch, kv_heads, sequence, head_dim] for both K and V.

    This is the cache tensor payload only, not allocator overhead.
    """
    cfg = model.config
    n_heads = cfg.n_head
    head_dim = cfg.n_embd // n_heads
    n_kv_heads = 1 if getattr(cfg, "mqa", False) else n_heads
    bytes_per_element = torch.tensor([], dtype=cache_dtype).element_size()

    num_elements = (
        2  # keys and values
        * cfg.n_layer
        * batch_size
        * n_kv_heads
        * sequence_length
        * head_dim
    )
    return {
        "kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "sequence_length": sequence_length,
        "estimated_kv_cache_bytes": num_elements * bytes_per_element,
        "estimated_kv_cache_mib": num_elements * bytes_per_element / (1024 ** 2),
    }


@torch.no_grad()
def benchmark_inference(
    model,
    data,
    device,
    amp_dtype,
    use_amp,
    prompt_length=128,
    gen_tokens=100,
    warmup=5,
    repeats=20,
):
    if prompt_length <= 0 or gen_tokens <= 0:
        raise ValueError("prompt_length and gen_tokens must both be positive.")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0.")
    if prompt_length + gen_tokens > model.config.block_size:
        raise ValueError(
            "prompt_length + gen_tokens must not exceed model block_size."
        )
    if len(data) < prompt_length:
        raise ValueError("Dataset does not contain enough tokens for the prompt.")

    model.eval()
    prompt = torch.as_tensor(
        np.asarray(data[:prompt_length], dtype=np.int64),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    autocast_context = (
        lambda: torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        )
    )

    def generate_once():
        with autocast_context():
            return model.generate(prompt, max_new_tokens=gen_tokens)

    # Warm up kernels and allocator. Warmups are excluded from measurements.
    for _ in range(warmup):
        _ = generate_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_times = []
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    baseline_allocated_bytes = 0

    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
            baseline_allocated_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()
        generated = generate_once()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_times.append(time.perf_counter() - start_time)
        del generated

        if device.type == "cuda":
            peak_allocated_bytes = max(
                peak_allocated_bytes, torch.cuda.max_memory_allocated(device)
            )
            peak_reserved_bytes = max(
                peak_reserved_bytes, torch.cuda.max_memory_reserved(device)
            )

    average_seconds = sum(elapsed_times) / len(elapsed_times)
    ms_per_token = average_seconds * 1000.0 / gen_tokens
    tokens_per_second = gen_tokens / average_seconds

    # The cache grows to approximately prompt + generated length.
    # Actual implementation may differ by one token depending on when it
    # appends the final generated token.
    parameter_dtype = next(model.parameters()).dtype
    cache_dtype = amp_dtype if use_amp else parameter_dtype
    cache_estimate = theoretical_kv_cache_size(
        model,
        batch_size=1,
        sequence_length=prompt_length + gen_tokens,
        cache_dtype=cache_dtype,
    )

    if device.type == "cuda":
        peak_increment_bytes = max(0, peak_allocated_bytes - baseline_allocated_bytes)
        memory_results = {
            "gpu_peak_allocated_bytes": peak_allocated_bytes,
            "gpu_peak_allocated_mib": peak_allocated_bytes / (1024 ** 2),
            "gpu_peak_reserved_bytes": peak_reserved_bytes,
            "gpu_peak_reserved_mib": peak_reserved_bytes / (1024 ** 2),
            "gpu_peak_increment_over_baseline_bytes": peak_increment_bytes,
            "gpu_peak_increment_over_baseline_mib": peak_increment_bytes / (1024 ** 2),
        }
    else:
        memory_results = {
            "gpu_memory_tracking": "unavailable (CPU execution)",
        }

    results = {
        "prompt_length": prompt_length,
        "generated_tokens": gen_tokens,
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "average_generation_time_seconds": average_seconds,
        "ms_per_token": ms_per_token,
        "tokens_per_second": tokens_per_second,
        "kv_cache_enabled": bool(getattr(model.config, "kv_cache", False)),
        "kv_cache_estimate": cache_estimate,
        "memory": memory_results,
    }

    print("\n===== Inference benchmark =====")
    print(f"Prompt length:              {prompt_length} tokens")
    print(f"Generated tokens:           {gen_tokens}")
    print(f"Average generation time:    {average_seconds:.4f} s")
    print(f"Average time per token:     {ms_per_token:.3f} ms")
    print(f"Generation speed:           {tokens_per_second:.2f} tokens/s")
    print(f"KV heads:                   {cache_estimate['kv_heads']}")
    print(
        "Estimated KV cache:         "
        f"{cache_estimate['estimated_kv_cache_mib']:.3f} MiB"
    )

    if device.type == "cuda":
        print(
            "Peak allocated GPU memory: "
            f"{memory_results['gpu_peak_allocated_mib']:.2f} MiB"
        )
        print(
            "Peak reserved GPU memory:  "
            f"{memory_results['gpu_peak_reserved_mib']:.2f} MiB"
        )
        print(
            "Peak increase over baseline:"
            f" {memory_results['gpu_peak_increment_over_baseline_mib']:.2f} MiB"
        )
    else:
        print("GPU memory tracking:        unavailable (CPU execution)")

    return results


def evaluate():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    amp_dtype, use_amp = get_autocast_settings(device, args.dtype)

    checkpoint_path = os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = dict(checkpoint["model_args"])

    # Important: mqa must be saved correctly in checkpoint["model_args"].
    # Do not silently treat a missing MQA flag as proof that the checkpoint is MHA.
    if "mqa" not in model_args:
        raise KeyError(
            "The checkpoint's model_args has no 'mqa' field. "
            "Check train.py: make sure mqa is included in model_args when "
            "the model is created and when the checkpoint is saved."
        )

    # kv_cache is an inference setting, not a learned model parameter.
    # It must be supported by your GPTConfig/model implementation.
    model_args["kv_cache"] = args.kv_cache

    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    num_parameters = sum(p.numel() for p in model.parameters())

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Dataset:           {args.dataset}")
    print(f"Attention:         {'MQA' if config.mqa else 'MHA'}")
    print(f"KV cache:          {'enabled' if args.kv_cache else 'disabled'}")
    print(f"Parameters:        {num_parameters:,}")
    print(f"Device:            {device}")
    print(f"Precision:         {args.dtype}")

    data_path = os.path.join("data", args.dataset, f"{args.split}.bin")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    block_size = config.block_size

    num_chunks = (len(data) - 1) // block_size
    if num_chunks == 0:
        raise ValueError("Dataset is too small for the model's block_size.")

    total_loss = 0.0
    total_tokens = 0

    print(f"\nEvaluating {num_chunks * block_size:,} tokens...")

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
                enabled=use_amp,
            ):
                logits, _, _ = model(x, y)

            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss += loss_sum.item()
            total_tokens += y.numel()

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "attention": "MQA" if config.mqa else "MHA",
        "kv_cache_enabled_for_benchmark": args.kv_cache,
        "parameters": num_parameters,
        "tokens_evaluated": total_tokens,
        "loss": average_loss,
        "perplexity": perplexity,
        "inference": None,
    }

    print("\n===== Evaluation results =====")
    print(f"Dataset:          {results['dataset']}")
    print(f"Split:            {results['split']}")
    print(f"Attention:        {results['attention']}")
    print(f"Parameters:       {results['parameters']:,}")
    print(f"Tokens evaluated: {results['tokens_evaluated']:,}")
    print(f"Loss:             {results['loss']:.4f}")
    print(f"Perplexity:       {results['perplexity']:.4f}")

    if args.benchmark_inference:
        results["inference"] = benchmark_inference(
            model=model,
            data=data,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            prompt_length=args.prompt_length,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    if args.save_json:
        os.makedirs(args.out_dir, exist_ok=True)
        output_path = os.path.join(
            args.out_dir,
            f"evaluation_{args.split}_{'kv' if args.kv_cache else 'no_kv'}.json",
        )
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=4)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    evaluate()

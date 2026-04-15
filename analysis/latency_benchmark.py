#!/usr/bin/env python3
"""Latency and memory benchmark for VLM adapter inference.

Measures prefill latency, decode throughput, and peak GPU memory
for different T (visual token count) and LLM scales.

Usage:
    python analysis/latency_benchmark.py [--llms 3B 14B] [--tokens 32 64 128]
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import VLMForScaling


def measure_prefill(model, pixel_values, prompt_ids, prompt_mask, n_warmup=3, n_runs=10):
    """Measure prefill latency: image encoding + adapter + LLM first forward."""
    device = model.device

    # Warmup
    for _ in range(n_warmup):
        vision_features = model.encode_vision(pixel_values.to(device))
        visual_tokens = model.adapter(vision_features.to(model.torch_dtype))
        B, T, _ = visual_tokens.shape
        embed_device = next(model.llm.get_input_embeddings().parameters()).device
        visual_embeds = visual_tokens.to(device=embed_device, dtype=model.torch_dtype)
        text_embeds = model.llm.get_input_embeddings()(prompt_ids.to(embed_device))
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        vis_mask = torch.ones(B, T, dtype=torch.long, device=embed_device)
        attn_mask = torch.cat([vis_mask, prompt_mask.to(embed_device)], dim=1)
        _ = model.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        vision_features = model.encode_vision(pixel_values.to(device))
        visual_tokens = model.adapter(vision_features.to(model.torch_dtype))
        B, T, _ = visual_tokens.shape
        embed_device = next(model.llm.get_input_embeddings().parameters()).device
        visual_embeds = visual_tokens.to(device=embed_device, dtype=model.torch_dtype)
        text_embeds = model.llm.get_input_embeddings()(prompt_ids.to(embed_device))
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        vis_mask = torch.ones(B, T, dtype=torch.long, device=embed_device)
        attn_mask = torch.cat([vis_mask, prompt_mask.to(embed_device)], dim=1)
        _ = model.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


def measure_generate(model, pixel_values, prompt_ids, prompt_mask,
                     max_new_tokens=64, n_warmup=2, n_runs=5):
    """Measure end-to-end generation (prefill + decode)."""
    # Warmup
    for _ in range(n_warmup):
        _ = model.generate(pixel_values, prompt_ids, prompt_mask,
                           max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    n_tokens_list = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model.generate(pixel_values, prompt_ids, prompt_mask,
                                 max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_tokens_list.append(outputs.shape[1])

    return times, n_tokens_list


def run_benchmark(llm_name, num_queries, adapter_level="M", batch_size=1):
    """Run full benchmark for one configuration."""
    print(f"\n{'='*60}")
    print(f"  LLM={llm_name}, T={num_queries}, adapter={adapter_level}")
    print(f"{'='*60}")

    # Build model (random adapter weights - fine for latency measurement)
    model = VLMForScaling(
        llm_name=llm_name,
        vision_name="google/siglip-so400m-patch14-384",
        adapter_level=adapter_level,
        num_queries=num_queries,
        adapter_num_layers=2,
        adapter_type="perceiver",
        torch_dtype=torch.bfloat16,
    )
    model.load_backbones(device="cuda")
    model.adapter = model.adapter.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    meta = model.get_vision_metadata()
    params = model.get_param_summary()
    print(f"  N_A={meta['N_A']:,}, T={meta['T']}")
    print(f"  LLM params: {params['llm']:,}")

    # Create dummy inputs
    pixel_values = torch.randn(batch_size, 3, 384, 384, dtype=torch.bfloat16, device="cuda")
    prompt_ids = torch.ones(batch_size, 32, dtype=torch.long, device="cuda")
    prompt_mask = torch.ones(batch_size, 32, dtype=torch.long, device="cuda")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # 1. Prefill latency
    print("  Measuring prefill...")
    with torch.no_grad():
        prefill_times = measure_prefill(model, pixel_values, prompt_ids, prompt_mask)
    prefill_median = sorted(prefill_times)[len(prefill_times) // 2]
    prefill_mean = sum(prefill_times) / len(prefill_times)

    # 2. Generation (prefill + decode)
    print("  Measuring generation...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        gen_times, n_tokens = measure_generate(model, pixel_values, prompt_ids, prompt_mask)
    gen_median = sorted(gen_times)[len(gen_times) // 2]
    gen_mean = sum(gen_times) / len(gen_times)
    avg_tokens = sum(n_tokens) / len(n_tokens)

    # 3. Peak memory
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Decode throughput (tokens/sec, excluding prefill)
    decode_time = gen_median - prefill_median
    decode_tok_per_sec = (avg_tokens / decode_time) if decode_time > 0 else float("inf")

    result = {
        "llm": llm_name.split("/")[-1],
        "T": num_queries,
        "adapter_level": adapter_level,
        "N_A": meta["N_A"],
        "prefill_ms": round(prefill_median * 1000, 1),
        "generation_ms": round(gen_median * 1000, 1),
        "decode_tok_per_sec": round(decode_tok_per_sec, 1),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "avg_generated_tokens": round(avg_tokens, 1),
    }

    print(f"  Prefill: {result['prefill_ms']}ms")
    print(f"  Generation (64 tok): {result['generation_ms']}ms")
    print(f"  Decode: {result['decode_tok_per_sec']} tok/s")
    print(f"  Peak memory: {result['peak_memory_gb']} GB")

    # Cleanup
    del model, pixel_values, prompt_ids, prompt_mask
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llms", nargs="+", default=["3B", "14B"],
                        help="LLM sizes to benchmark")
    parser.add_argument("--tokens", nargs="+", type=int, default=[32, 64, 128],
                        help="T (num_queries) values to sweep")
    parser.add_argument("--adapter_level", default="M")
    parser.add_argument("--output", default="analysis/latency_results.json")
    args = parser.parse_args()

    llm_map = {
        "1.5B": "Qwen/Qwen2.5-1.5B",
        "3B": "Qwen/Qwen2.5-3B",
        "7B": "Qwen/Qwen2.5-7B",
        "14B": "Qwen/Qwen2.5-14B",
    }

    results = []
    for llm_size in args.llms:
        llm_name = llm_map[llm_size]
        for T in args.tokens:
            result = run_benchmark(llm_name, T, args.adapter_level)
            results.append(result)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'LLM':<12} {'T':>4} {'Prefill(ms)':>12} {'Gen(ms)':>10} {'Tok/s':>8} {'Mem(GB)':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['llm']:<12} {r['T']:>4} {r['prefill_ms']:>12} {r['generation_ms']:>10} "
              f"{r['decode_tok_per_sec']:>8} {r['peak_memory_gb']:>8}")


if __name__ == "__main__":
    main()

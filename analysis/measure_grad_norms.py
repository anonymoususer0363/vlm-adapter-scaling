"""
Measure gradient norms during early training across LR × N_A × N_L.

Purpose: Provide mechanistic evidence for the hook-shaped N_A curve.
  - At LR=3e-4, large adapters produce large gradient norms → instability
  - At LR=1e-4, gradient norms stay manageable → monotonic improvement

Usage:
    # Full sweep (4 LLM × 5 N_A × 2 LR = 40 combos, ~6h on A6000×2)
    python analysis/measure_grad_norms.py

    # Quick test (3B only)
    python analysis/measure_grad_norms.py --llms 3B --steps 50

    # Custom sweep
    python analysis/measure_grad_norms.py --llms 3B 7B 14B --adapters XS M XL --lrs 1e-4 3e-4

    # Plot only (from saved results)
    python analysis/measure_grad_norms.py --plot_only --results_path analysis/grad_norm_results.json
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import VLMForScaling, QWEN25_CONFIGS
from src.data import build_dataloader


# Adapter hidden dims (from perceiver_resampler.py ADAPTER_CONFIGS)
ADAPTER_LEVELS = {
    "XS": {"d": 512},
    "S": {"d": 768},
    "M": {"d": 1024},
    "L": {"d": 1280},
    "XL": {"d": 1536},
}

LLM_SIZES = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-1.5B",
    "3B": "Qwen/Qwen2.5-3B",
    "7B": "Qwen/Qwen2.5-7B",
    "14B": "Qwen/Qwen2.5-14B",
}


def compute_grad_norm(model, norm_type=2.0):
    """Compute total gradient L2 norm of adapter parameters (before clipping)."""
    params = [p for p in model.adapter.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type,
    )
    return total_norm.item()


def compute_param_norm(model, norm_type=2.0):
    """Compute total parameter L2 norm of adapter."""
    params = [p for p in model.adapter.parameters() if p.requires_grad]
    if not params:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type,
    )
    return total_norm.item()


def run_one_config(llm_name, adapter_level, lr, num_steps, data_path, val_data,
                   image_root, batch_size, seed, device, dtype):
    """Run short training and record gradient norms."""
    print(f"\n{'='*60}")
    print(f"Config: LLM={llm_name}, N_A={adapter_level}, LR={lr}")
    print(f"{'='*60}")

    # Build model
    model = VLMForScaling(
        llm_name=llm_name,
        adapter_level=adapter_level,
        num_queries=64,  # fixed T=64 (canonical)
        adapter_num_layers=2,
        torch_dtype=dtype,
    )
    model.adapter = model.adapter.to(device=device, dtype=dtype)
    model.load_backbones(device=device)

    n_adapter_params = model.adapter.num_params
    print(f"  Adapter params: {n_adapter_params:,}")

    # Build data loader
    train_loader = build_dataloader(
        data_path=data_path,
        image_root=image_root,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        num_samples=num_steps * batch_size * 2,  # enough for num_steps
        num_workers=4,
        max_text_len=512,
        seed=seed,
    )

    # Optimizer (same as real training)
    optimizer = torch.optim.AdamW(
        [p for p in model.adapter.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Measure
    model.adapter.train()
    results = {
        "steps": [],
        "grad_norms": [],
        "update_norms": [],  # LR * grad_norm (effective step size)
        "param_norms": [],
        "losses": [],
        "grad_norm_per_layer": [],  # per-layer breakdown
    }

    initial_param_norm = compute_param_norm(model)
    results["initial_param_norm"] = initial_param_norm

    data_iter = iter(train_loader)
    t0 = time.time()

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        # Forward
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = outputs["loss"]
        loss.backward()

        # Measure gradient norm BEFORE clipping
        grad_norm = compute_grad_norm(model)
        param_norm = compute_param_norm(model)
        update_norm = lr * grad_norm  # approximate effective step size

        # Per-layer gradient norms
        layer_norms = {}
        for name, p in model.adapter.named_parameters():
            if p.requires_grad and p.grad is not None:
                layer_norms[name] = torch.norm(p.grad.detach(), 2.0).item()

        results["steps"].append(step)
        results["grad_norms"].append(grad_norm)
        results["update_norms"].append(update_norm)
        results["param_norms"].append(param_norm)
        results["losses"].append(loss.item())
        results["grad_norm_per_layer"].append(layer_norms)

        # Clip and step (like real training)
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}, "
                  f"grad_norm={grad_norm:.4f}, update_norm={update_norm:.6f}, "
                  f"param_norm={param_norm:.2f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Summary stats
    gn = np.array(results["grad_norms"])
    results["summary"] = {
        "mean_grad_norm": float(np.mean(gn)),
        "std_grad_norm": float(np.std(gn)),
        "max_grad_norm": float(np.max(gn)),
        "median_grad_norm": float(np.median(gn)),
        "mean_update_norm": float(np.mean(results["update_norms"])),
        "mean_loss": float(np.mean(results["losses"])),
        "final_loss": float(results["losses"][-1]) if results["losses"] else float("nan"),
        "n_adapter_params": n_adapter_params,
        "elapsed_sec": elapsed,
    }

    # Cleanup to free GPU memory
    del model, optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    return results


def plot_grad_norms(all_results, output_dir):
    """Generate gradient norm figures for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(output_dir, exist_ok=True)

    # Color scheme
    adapter_colors = {
        "XS": "#2196F3",  # blue
        "S": "#4CAF50",   # green
        "M": "#FF9800",   # orange
        "L": "#9C27B0",   # purple
        "XL": "#F44336",  # red
    }
    lr_styles = {
        "1e-4": "--",   # dashed
        "1e-04": "--",
        "3e-4": "-",    # solid
        "3e-04": "-",
        "0.0001": "--",
        "0.0003": "-",
    }

    # Normalize LR keys
    def norm_lr(lr_str):
        lr_val = float(lr_str)
        if abs(lr_val - 1e-4) < 1e-6:
            return "1e-4"
        elif abs(lr_val - 3e-4) < 1e-6:
            return "3e-4"
        return lr_str

    # ---- Figure 1: Grad norm vs step, one panel per LLM ----
    llms = sorted(set(r["llm"] for r in all_results))
    lrs = sorted(set(norm_lr(r["lr"]) for r in all_results))
    adapters = sorted(set(r["adapter"] for r in all_results),
                      key=lambda x: list(ADAPTER_LEVELS.keys()).index(x))

    if len(llms) > 1:
        fig, axes = plt.subplots(1, len(llms), figsize=(4.5 * len(llms), 3.5), sharey=True)
        if len(llms) == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 3.5))
        axes = [axes]

    for ax, llm in zip(axes, llms):
        for r in all_results:
            if r["llm"] != llm:
                continue
            adapter = r["adapter"]
            lr_key = norm_lr(r["lr"])
            steps = r["data"]["steps"]
            gn = r["data"]["grad_norms"]
            style = lr_styles.get(lr_key, "-")
            color = adapter_colors.get(adapter, "gray")
            ax.plot(steps, gn, style, color=color, alpha=0.8, linewidth=1.5)

        ax.set_title(f"Qwen2.5-{llm}", fontsize=11)
        ax.set_xlabel("Training Step", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Gradient Norm (pre-clip)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = []
    for adapter in adapters:
        if adapter in adapter_colors:
            legend_elements.append(
                Line2D([0], [0], color=adapter_colors[adapter], lw=2, label=f"N_A={adapter}")
            )
    for lr_key in lrs:
        style = lr_styles.get(lr_key, "-")
        legend_elements.append(
            Line2D([0], [0], color="gray", lw=1.5, linestyle=style, label=f"LR={lr_key}")
        )
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(adapters) + len(lrs),
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{output_dir}/grad_norm_vs_step.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"{output_dir}/grad_norm_vs_step.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/grad_norm_vs_step.pdf")

    # ---- Figure 2: Mean grad norm vs N_A, lines for LR, panels for LLM ----
    # This is the key figure for the paper
    if len(llms) > 1:
        fig, axes = plt.subplots(1, len(llms), figsize=(4.5 * len(llms), 3.5), sharey=True)
        if len(llms) == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 3.5))
        axes = [axes]

    for ax, llm in zip(axes, llms):
        for lr_key in lrs:
            na_sizes = []
            mean_gn = []
            std_gn = []
            adapter_labels = []
            for adapter in adapters:
                matching = [r for r in all_results
                           if r["llm"] == llm and r["adapter"] == adapter
                           and norm_lr(r["lr"]) == lr_key]
                if matching:
                    r = matching[0]
                    na_sizes.append(r["data"]["summary"]["n_adapter_params"])
                    mean_gn.append(r["data"]["summary"]["mean_grad_norm"])
                    std_gn.append(r["data"]["summary"]["std_grad_norm"])
                    adapter_labels.append(adapter)

            if na_sizes:
                style = lr_styles.get(lr_key, "-")
                ax.errorbar(na_sizes, mean_gn, yerr=std_gn,
                           fmt="o" + style, color="tab:red" if "3e-4" in lr_key else "tab:blue",
                           linewidth=1.5, markersize=5, capsize=3, alpha=0.8,
                           label=f"LR={lr_key}")
                # Label adapter sizes
                for x, y, label in zip(na_sizes, mean_gn, adapter_labels):
                    ax.annotate(label, (x, y), textcoords="offset points",
                               xytext=(0, 8), ha="center", fontsize=7, alpha=0.7)

        ax.set_title(f"Qwen2.5-{llm}", fontsize=11)
        ax.set_xlabel("Adapter Parameters (N_A)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Mean Gradient Norm", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/grad_norm_vs_NA.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"{output_dir}/grad_norm_vs_NA.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/grad_norm_vs_NA.pdf")

    # ---- Figure 3: Update norm (LR × grad_norm) vs N_A ----
    # This directly shows "effective step size" — the mechanism behind instability
    if len(llms) > 1:
        fig, axes = plt.subplots(1, len(llms), figsize=(4.5 * len(llms), 3.5), sharey=True)
        if len(llms) == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 3.5))
        axes = [axes]

    for ax, llm in zip(axes, llms):
        for lr_key in lrs:
            na_sizes = []
            mean_un = []
            adapter_labels = []
            for adapter in adapters:
                matching = [r for r in all_results
                           if r["llm"] == llm and r["adapter"] == adapter
                           and norm_lr(r["lr"]) == lr_key]
                if matching:
                    r = matching[0]
                    na_sizes.append(r["data"]["summary"]["n_adapter_params"])
                    mean_un.append(r["data"]["summary"]["mean_update_norm"])
                    adapter_labels.append(adapter)

            if na_sizes:
                color = "tab:red" if "3e-4" in lr_key else "tab:blue"
                style = lr_styles.get(lr_key, "-")
                ax.plot(na_sizes, mean_un, "o" + style, color=color,
                       linewidth=1.5, markersize=5, alpha=0.8, label=f"LR={lr_key}")
                for x, y, label in zip(na_sizes, mean_un, adapter_labels):
                    ax.annotate(label, (x, y), textcoords="offset points",
                               xytext=(0, 8), ha="center", fontsize=7, alpha=0.7)

        ax.set_title(f"Qwen2.5-{llm}", fontsize=11)
        ax.set_xlabel("Adapter Parameters (N_A)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Update Norm (LR × ‖∇‖)", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/update_norm_vs_NA.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"{output_dir}/update_norm_vs_NA.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/update_norm_vs_NA.pdf")

    # ---- Print summary table ----
    print(f"\n{'='*80}")
    print("Summary: Mean Gradient Norm (± std)")
    print(f"{'='*80}")
    header = f"{'LLM':>6} {'N_A':>4} {'LR':>6} {'Mean GN':>10} {'Std GN':>10} {'Update':>10} {'Params':>12}"
    print(header)
    print("-" * len(header))
    for r in sorted(all_results, key=lambda x: (x["llm"], x["adapter"], x["lr"])):
        s = r["data"]["summary"]
        print(f"{r['llm']:>6} {r['adapter']:>4} {norm_lr(r['lr']):>6} "
              f"{s['mean_grad_norm']:>10.4f} {s['std_grad_norm']:>10.4f} "
              f"{s['mean_update_norm']:>10.6f} {s['n_adapter_params']:>12,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llms", nargs="+", default=["1.5B", "3B", "7B", "14B"])
    parser.add_argument("--adapters", nargs="+", default=["XS", "S", "M", "L", "XL"])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--steps", type=int, default=100, help="Steps per config")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--image_root", type=str, default="data/llava_pretrain")
    parser.add_argument("--output_dir", type=str, default="analysis/grad_norms")
    parser.add_argument("--results_path", type=str, default="analysis/grad_norms/results.json")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.plot_only:
        print(f"Loading results from {args.results_path}")
        with open(args.results_path) as f:
            all_results = json.load(f)
        plot_grad_norms(all_results, output_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    total_combos = len(args.llms) * len(args.adapters) * len(args.lrs)
    print(f"Gradient Norm Measurement: {total_combos} combinations")
    print(f"  LLMs: {args.llms}")
    print(f"  Adapters: {args.adapters}")
    print(f"  LRs: {args.lrs}")
    print(f"  Steps: {args.steps}")

    all_results = []
    combo_idx = 0

    for llm in args.llms:
        llm_name = LLM_SIZES[llm]
        # Adjust batch size for large models
        bs = args.batch_size
        if llm in ("14B", "32B"):
            bs = min(bs, 4)

        for adapter in args.adapters:
            for lr in args.lrs:
                combo_idx += 1
                lr_str = f"{lr:.0e}"
                print(f"\n[{combo_idx}/{total_combos}] LLM={llm}, N_A={adapter}, LR={lr_str}")

                try:
                    data = run_one_config(
                        llm_name=llm_name,
                        adapter_level=adapter,
                        lr=lr,
                        num_steps=args.steps,
                        data_path=args.train_data,
                        val_data=args.val_data,
                        image_root=args.image_root,
                        batch_size=bs,
                        seed=args.seed,
                        device=device,
                        dtype=dtype,
                    )

                    # Store (drop per-layer norms to save space in main results)
                    result_entry = {
                        "llm": llm,
                        "adapter": adapter,
                        "lr": str(lr),
                        "data": {
                            "steps": data["steps"],
                            "grad_norms": data["grad_norms"],
                            "update_norms": data["update_norms"],
                            "param_norms": data["param_norms"],
                            "losses": data["losses"],
                            "initial_param_norm": data["initial_param_norm"],
                            "summary": data["summary"],
                        },
                    }
                    all_results.append(result_entry)

                    # Save incrementally
                    with open(args.results_path, "w") as f:
                        json.dump(all_results, f, indent=2)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"\nAll measurements complete. Results saved to {args.results_path}")

    # Plot
    plot_grad_norms(all_results, output_dir)


if __name__ == "__main__":
    main()

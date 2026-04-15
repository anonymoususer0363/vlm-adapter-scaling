"""
Main training script for VLM adapter scaling law experiments.

Usage:
    python train.py --config configs/g0/n0.5b_d5m.yaml

    # Or with command-line overrides:
    python train.py \
        --llm_name Qwen/Qwen2.5-3B \
        --adapter_level M \
        --num_queries 64 \
        --num_samples 50000000 \
        --batch_size 32 \
        --run_name g0_n3b_d50m
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

# Auto-detect local model cache (for B200 transfer)
_project_root = Path(__file__).parent
_local_cache = _project_root / "hf_cache"
if _local_cache.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_local_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(_local_cache / "hub")

from src.model import VLMForScaling
from src.data import build_dataloader
from src.trainer import Trainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Adapter Scaling Law Training")

    # Config file (overrides defaults, overridden by CLI args)
    parser.add_argument("--config", type=str, default=None, help="YAML config file")

    # Model
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--vision_name", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--adapter_level", type=str, default="M", choices=["XS", "S", "M", "L", "XL"])
    parser.add_argument("--num_queries", type=int, default=64, help="T: number of visual tokens")
    parser.add_argument("--adapter_num_layers", type=int, default=2)
    parser.add_argument("--adapter_type", type=str, default="perceiver",
                        choices=["perceiver", "mlp", "qformer"],
                        help="Adapter architecture type (G11)")

    # Data
    parser.add_argument("--train_data", type=str, required=False, help="Path to training data file")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data file")
    parser.add_argument("--image_root", type=str, default="", help="Root dir for images")
    parser.add_argument("--num_samples", type=int, default=None, help="D: number of training samples")
    parser.add_argument("--max_text_len", type=int, default=512)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)

    # Logging & saving
    parser.add_argument("--output_dir", type=str, default=os.environ.get("VLM_CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_interval_steps", type=int, default=500)
    parser.add_argument("--save_interval_steps", type=int, default=1000)
    parser.add_argument("--log_interval_steps", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="vlm-adapter-scaling")

    # LoRA (for G6: LLM unfrozen)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    # D prefix-reuse: save checkpoints at these seen_pairs milestones
    parser.add_argument("--save_at_milestones", type=int, nargs="*", default=None,
                        help="seen_pairs milestones for D prefix-reuse (e.g., 5000000 20000000 50000000)")

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Config file values override defaults but not CLI args
        for key, value in config.items():
            if key in vars(args) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    # Auto-generate run name
    if args.run_name is None:
        llm_short = args.llm_name.split("/")[-1].replace("Qwen2.5-", "")
        d_short = f"d{args.num_samples // 1_000_000}m" if args.num_samples else "dall"
        args.run_name = f"{llm_short}_T{args.num_queries}_{args.adapter_level}_{d_short}_s{args.seed}"

    return args


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Dtype: {args.dtype}")

    # Build model
    adapter_type = getattr(args, "adapter_type", "perceiver")
    print(f"Building model: LLM={args.llm_name}, Adapter={args.adapter_level}, "
          f"T={args.num_queries}, type={adapter_type}")
    model = VLMForScaling(
        llm_name=args.llm_name,
        vision_name=args.vision_name,
        adapter_level=args.adapter_level,
        num_queries=args.num_queries,
        adapter_num_layers=args.adapter_num_layers,
        adapter_type=adapter_type,
        torch_dtype=torch_dtype,
    )
    model.adapter = model.adapter.to(device=device, dtype=torch_dtype)
    model.load_backbones(device=device)

    # Apply LoRA if requested (G6: LLM unfrozen)
    if args.use_lora:
        print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        model.apply_lora(r=args.lora_r, alpha=args.lora_alpha)

    param_summary = model.get_param_summary()
    print(f"Parameters:")
    print(f"  Adapter (trainable): {param_summary['adapter_trainable']:>12,}")
    print(f"  Vision Encoder:      {param_summary['vision_encoder']:>12,}")
    print(f"  LLM:                 {param_summary['llm']:>12,}")
    print(f"  Total:               {param_summary['total']:>12,}")
    print(f"  Trainable ratio:     {param_summary['trainable_ratio']:.4%}")

    # Build dataloaders
    print(f"Loading data: {args.train_data}, D={args.num_samples}")
    train_loader = build_dataloader(
        data_path=args.train_data,
        image_root=args.image_root,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        max_text_len=args.max_text_len,
        seed=args.seed,
    )

    val_loader = None
    if args.val_data:
        val_loader = build_dataloader(
            data_path=args.val_data,
            image_root=args.image_root,
            image_processor=model.image_processor,
            tokenizer=model.tokenizer,
            batch_size=args.batch_size,
            num_samples=None,
            num_workers=args.num_workers,
            max_text_len=args.max_text_len,
            seed=args.seed,
        )

    # Vision metadata (T₀, ρ, etc.)
    vision_metadata = model.get_vision_metadata()
    print(f"Vision metadata: T₀={vision_metadata['T0']}, T={vision_metadata['T']}, "
          f"ρ={vision_metadata['rho']:.4f}, N_A={vision_metadata['N_A']:,}")

    # Trainer config
    train_config = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        eval_interval_steps=args.eval_interval_steps,
        save_interval_steps=args.save_interval_steps,
        log_interval_steps=args.log_interval_steps,
        save_at_milestones=args.save_at_milestones,
        output_dir=args.output_dir,
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Save full config (including vision metadata)
    config_save_dir = Path(args.output_dir) / args.run_name
    config_save_dir.mkdir(parents=True, exist_ok=True)
    with open(config_save_dir / "config.json", "w") as f:
        json.dump({**vars(args), **param_summary, **vision_metadata}, f, indent=2, default=str)

    # Train
    trainer = Trainer(model, train_loader, val_loader, train_config, vision_metadata=vision_metadata)
    result = trainer.train()
    print(f"Done. Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()

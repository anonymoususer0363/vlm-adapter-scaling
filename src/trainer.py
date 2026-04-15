"""
Training loop for VLM adapter scaling law experiments.

Features:
- Adapter-only training (VE + LLM frozen)
- Gradient accumulation
- Cosine LR schedule with warmup
- Validation NLL logging
- Checkpoint saving
- WandB logging
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.0

    # Training
    num_epochs: int = 1
    batch_size: int = 32
    grad_accum_steps: int = 1
    eval_interval_steps: int = 500
    save_interval_steps: int = 1000
    log_interval_steps: int = 50

    # D prefix-reuse: save checkpoint + eval at these seen_pairs milestones
    save_at_milestones: list[int] | None = None

    # Paths
    output_dir: str = "checkpoints"
    run_name: str = "default"

    # WandB
    use_wandb: bool = True
    wandb_project: str = "vlm-adapter-scaling"


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainConfig,
        vision_metadata: dict | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = model.device
        self.vision_metadata = vision_metadata or {}

        # Optimize adapter + LoRA parameters (if applicable)
        trainable_params = model.trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            [p for p in trainable_params if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        # Compute total steps
        self.steps_per_epoch = len(train_loader) // config.grad_accum_steps
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        # Output dir
        self.save_dir = Path(config.output_dir) / config.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.global_step = 0
        self.seen_pairs = 0
        self.best_val_loss = float("inf")

        # D milestone tracking
        self._reached_milestones = set()
        self.milestone_results = {}  # milestone -> val_loss

        # WandB
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                wandb_config = {**vars(config)}
                if self.vision_metadata:
                    wandb_config.update({f"vision/{k}": v for k, v in self.vision_metadata.items()})
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.run_name,
                    config=wandb_config,
                    resume="allow",
                )
            except Exception as e:
                print(f"WandB init failed: {e}. Continuing without WandB.")

    def get_lr(self, step: int) -> float:
        """Cosine schedule with warmup."""
        if step < self.warmup_steps:
            return self.config.lr * step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = self.config.lr * self.config.min_lr_ratio
        return min_lr + (self.config.lr - min_lr) * cosine_decay

    def set_lr(self, step: int):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute validation NLL."""
        if self.val_loader is None:
            return float("nan")

        self.model.adapter.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_loader:
            outputs = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            # Weight by number of non-padding tokens
            num_tokens = (batch["attention_mask"].sum()).item()
            total_loss += outputs["loss"].item() * num_tokens
            total_tokens += num_tokens

        self.model.adapter.train()
        return total_loss / max(1, total_tokens)

    def save_checkpoint(self, tag: str = "latest"):
        """Save adapter checkpoint."""
        path = self.save_dir / f"adapter_{tag}.pt"
        torch.save({
            "adapter_state_dict": self.model.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)

    def train(self):
        """Main training loop."""
        cfg = self.config
        self.model.adapter.train()

        print(f"Starting training: {self.total_steps} steps, "
              f"{self.warmup_steps} warmup, "
              f"adapter params: {self.model.adapter.num_trainable_params:,}")

        accum_loss = 0.0
        accum_steps = 0
        start_time = time.time()

        for epoch in range(cfg.num_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                # Track seen pairs
                batch_sz = batch["pixel_values"].shape[0]
                self.seen_pairs += batch_sz

                # Forward
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = outputs["loss"] / cfg.grad_accum_steps
                loss.backward()

                accum_loss += loss.item()
                accum_steps += 1

                if accum_steps < cfg.grad_accum_steps:
                    continue

                # Gradient clipping
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.adapter.parameters(), cfg.grad_clip
                    )

                # Optimizer step
                lr = self.set_lr(self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                train_loss = accum_loss
                accum_loss = 0.0
                accum_steps = 0

                # Logging
                if self.global_step % cfg.log_interval_steps == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    eta_hours = (self.total_steps - self.global_step) / max(steps_per_sec, 1e-6) / 3600

                    log_dict = {
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/step": self.global_step,
                        "train/seen_pairs": self.seen_pairs,
                        "train/epoch": epoch + batch_idx / len(self.train_loader),
                        "train/steps_per_sec": steps_per_sec,
                        "train/eta_hours": eta_hours,
                    }
                    print(f"[Step {self.global_step}/{self.total_steps}] "
                          f"loss={train_loss:.4f} lr={lr:.2e} "
                          f"seen={self.seen_pairs:,} "
                          f"speed={steps_per_sec:.1f} step/s ETA={eta_hours:.1f}h")

                    if self.wandb_run:
                        import wandb
                        wandb.log(log_dict, step=self.global_step)

                # Evaluation
                if self.global_step % cfg.eval_interval_steps == 0:
                    val_loss = self.evaluate()
                    print(f"  [Eval] val_loss={val_loss:.4f}")

                    if self.wandb_run:
                        import wandb
                        wandb.log({"val/loss": val_loss}, step=self.global_step)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best")

                # D milestone checkpoints (prefix-reuse)
                if cfg.save_at_milestones:
                    for milestone in cfg.save_at_milestones:
                        if self.seen_pairs >= milestone and milestone not in self._reached_milestones:
                            self._reached_milestones.add(milestone)
                            val_loss = self.evaluate()
                            tag = f"d{milestone // 1_000_000}m"
                            self.save_checkpoint(tag)
                            self.milestone_results[milestone] = val_loss
                            print(f"  [Milestone D={milestone:,}] val_loss={val_loss:.4f} → saved {tag}")
                            if self.wandb_run:
                                import wandb
                                wandb.log({
                                    f"milestone/val_loss_d{milestone // 1_000_000}m": val_loss,
                                    "milestone/seen_pairs": self.seen_pairs,
                                }, step=self.global_step)

                # Save checkpoint
                if self.global_step % cfg.save_interval_steps == 0:
                    self.save_checkpoint("latest")

                if self.global_step >= self.total_steps:
                    break

            if self.global_step >= self.total_steps:
                break

        # Final eval and save
        val_loss = self.evaluate()
        print(f"Training complete. Final val_loss={val_loss:.4f}, best={self.best_val_loss:.4f}")
        self.save_checkpoint("final")

        # Save result summary
        result = {
            "run_name": cfg.run_name,
            "final_val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
            "seen_pairs": self.seen_pairs,
            "adapter_params": self.model.adapter.num_params,
            **{f"vision_{k}": v for k, v in self.vision_metadata.items()},
        }
        if self.milestone_results:
            result["milestone_val_losses"] = {
                f"d{m // 1_000_000}m": loss for m, loss in self.milestone_results.items()
            }
        result_path = self.save_dir / "result.json"
        import json
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        if self.wandb_run:
            import wandb
            wandb.finish()

        return result

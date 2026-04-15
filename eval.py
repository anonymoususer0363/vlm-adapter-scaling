"""
Evaluation script for G10 downstream correlation.

Loads a trained adapter checkpoint, runs inference on benchmarks,
and computes downstream metrics (VQA accuracy, CIDEr).

Usage:
    # Single checkpoint, single benchmark
    python eval.py --checkpoint checkpoints/g1_3B_T64_M_d50m_s42 --benchmark vqav2

    # Single checkpoint, all benchmarks
    python eval.py --checkpoint checkpoints/g1_3B_T64_M_d50m_s42 --benchmark all

    # Batch mode: evaluate multiple checkpoints (G10 config list)
    python eval.py --batch_config configs/g10_eval_list.txt --benchmark all

    # Quick test with few samples
    python eval.py --checkpoint checkpoints/g1_3B_T64_M_d50m_s42 --benchmark vqav2 --max_samples 100
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Auto-detect local model cache
_project_root = Path(__file__).parent
_local_cache = _project_root / "hf_cache"
if _local_cache.exists() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_local_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(_local_cache / "hub")

from src.model import VLMForScaling
from src.eval_datasets import (
    VQAv2Dataset, TextVQADataset, COCOCaptionDataset,
    build_eval_dataloader,
)
from src.metrics import compute_vqa_metrics, compute_caption_metrics


def load_model_from_checkpoint(checkpoint_dir: str, device: str = "cuda") -> VLMForScaling:
    """Load VLMForScaling model with trained adapter weights."""
    ckpt_path = Path(checkpoint_dir)
    config_path = ckpt_path / "config.json"
    adapter_path = ckpt_path / "adapter_best.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_best.pt not found in {ckpt_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Build model
    model = VLMForScaling(
        llm_name=config["llm_name"],
        vision_name=config.get("vision_name", "google/siglip-so400m-patch14-384"),
        adapter_level=config["adapter_level"],
        num_queries=config["num_queries"],
        adapter_num_layers=config.get("adapter_num_layers", 2),
        adapter_type=config.get("adapter_type", "perceiver"),
        torch_dtype=torch.bfloat16,
    )

    # Load adapter weights (checkpoint may wrap state_dict in a dict)
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if "adapter_state_dict" in checkpoint:
        state_dict = checkpoint["adapter_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.adapter.load_state_dict(state_dict)
    model.adapter = model.adapter.to(device=device, dtype=torch.bfloat16)

    # Load backbones
    model.load_backbones(device=device)

    model.eval()
    return model


@torch.no_grad()
def evaluate_vqa(
    model: VLMForScaling,
    dataset: VQAv2Dataset | TextVQADataset,
    batch_size: int = 8,
    max_new_tokens: int = 16,
) -> dict:
    """Run VQA evaluation."""
    loader = build_eval_dataloader(
        dataset,
        batch_size=batch_size,
        pad_token_id=model.tokenizer.pad_token_id or 0,
    )

    all_preds = []
    all_answers = []
    total = len(dataset)
    done = 0
    t0 = time.time()

    for batch in loader:
        outputs = model.generate(
            pixel_values=batch["pixel_values"],
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            max_new_tokens=max_new_tokens,
        )

        # Decode generated tokens (only newly generated, strip prompt if echoed)
        for i in range(outputs.shape[0]):
            gen_text = model.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            # If prompt is echoed in output, strip it
            if "prompt_texts" in batch["meta"] and i < len(batch["meta"]["prompt_texts"]):
                prompt_text = batch["meta"]["prompt_texts"][i]
                if gen_text.startswith(prompt_text):
                    gen_text = gen_text[len(prompt_text):].strip()
            all_preds.append(gen_text)

        all_answers.extend(batch["meta"]["answers"])
        done += outputs.shape[0]

        if done % (batch_size * 10) == 0 or done == total:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total}] {speed:.1f} samples/sec")

    metrics = compute_vqa_metrics(all_preds, all_answers)
    metrics["elapsed_sec"] = time.time() - t0
    return metrics


@torch.no_grad()
def evaluate_caption(
    model: VLMForScaling,
    dataset: COCOCaptionDataset,
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> dict:
    """Run caption evaluation."""
    loader = build_eval_dataloader(
        dataset,
        batch_size=batch_size,
        pad_token_id=model.tokenizer.pad_token_id or 0,
    )

    predictions = []
    references = []
    total = len(dataset)
    done = 0
    t0 = time.time()

    for batch in loader:
        outputs = model.generate(
            pixel_values=batch["pixel_values"],
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            max_new_tokens=max_new_tokens,
        )

        for i in range(outputs.shape[0]):
            gen_text = model.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            if "prompt_texts" in batch["meta"] and i < len(batch["meta"]["prompt_texts"]):
                prompt_text = batch["meta"]["prompt_texts"][i]
                if gen_text.startswith(prompt_text):
                    gen_text = gen_text[len(prompt_text):].strip()
            img_id = batch["meta"]["image_ids"][i]
            predictions.append({"image_id": img_id, "caption": gen_text})
            references.append({"image_id": img_id, "references": batch["meta"]["references"][i]})

        done += outputs.shape[0]
        if done % (batch_size * 10) == 0 or done == total:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total}] {speed:.1f} samples/sec")

    metrics = compute_caption_metrics(predictions, references)
    metrics["elapsed_sec"] = time.time() - t0
    return metrics


@torch.no_grad()
def compute_batch_answer_ppl(
    model: VLMForScaling,
    pixel_values: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    answer_ids: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample NLL of answer tokens given image + prompt.

    Builds sequence: [visual_tokens, prompt_embeds, answer_embeds]
    with labels only on answer token positions.

    Returns:
        (B,) tensor of mean NLL per sample.
    """
    B = pixel_values.shape[0]
    device = model.device

    # Vision encoding + adapter
    vision_features = model.encode_vision(pixel_values.to(device))
    visual_tokens = model.adapter(vision_features.to(model.torch_dtype))
    T = visual_tokens.shape[1]

    # Get embedding device (may differ from adapter device for multi-GPU)
    embed_device = next(model.llm.get_input_embeddings().parameters()).device

    # Build embeddings
    visual_embeds = visual_tokens.to(device=embed_device, dtype=model.torch_dtype)
    prompt_ids_dev = prompt_ids.to(embed_device)
    answer_ids_dev = answer_ids.to(embed_device)

    prompt_embeds = model.llm.get_input_embeddings()(prompt_ids_dev)
    answer_embeds = model.llm.get_input_embeddings()(answer_ids_dev)

    # Combined sequence: [visual, prompt, answer]
    inputs_embeds = torch.cat([visual_embeds, prompt_embeds, answer_embeds], dim=1)

    # Attention mask
    vis_mask = torch.ones(B, T, dtype=torch.long, device=embed_device)
    prompt_mask_dev = prompt_mask.to(embed_device)
    answer_mask_dev = answer_mask.to(embed_device)
    combined_mask = torch.cat([vis_mask, prompt_mask_dev, answer_mask_dev], dim=1)

    # Labels: -100 for visual + prompt, actual IDs for answer (masked padding)
    S_prompt = prompt_ids.shape[1]
    ignore_labels = torch.full((B, T + S_prompt), -100, dtype=torch.long, device=embed_device)
    answer_labels = answer_ids_dev.clone()
    answer_labels[answer_mask_dev == 0] = -100
    labels = torch.cat([ignore_labels, answer_labels], dim=1)

    # Forward through LLM
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=combined_mask,
        use_cache=False,
    )

    # Per-sample NLL computation
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(B, -1)

    # Average over answer tokens only
    answer_token_mask = (shift_labels != -100).float()
    per_sample_nll = (per_token_loss * answer_token_mask).sum(dim=1) / answer_token_mask.sum(dim=1).clamp(min=1)

    return per_sample_nll


@torch.no_grad()
def evaluate_vqa_ppl(
    model: VLMForScaling,
    dataset: VQAv2Dataset | TextVQADataset,
    batch_size: int = 8,
) -> dict:
    """
    PPL-based VQA evaluation.

    For each question, computes NLL of the most common ground-truth answer
    given image + question prompt. Returns mean NLL as a continuous metric
    that can be correlated with pretraining val loss.
    """
    loader = build_eval_dataloader(
        dataset,
        batch_size=batch_size,
        pad_token_id=model.tokenizer.pad_token_id or 0,
    )

    all_nlls = []
    total = len(dataset)
    done = 0
    t0 = time.time()

    for batch in loader:
        B = batch["pixel_values"].shape[0]
        answers_list = batch["meta"]["answers"]  # list of list[str]

        # Get most common answer per sample
        best_answers = []
        for answers in answers_list:
            counts = Counter(answers)
            best_answers.append(counts.most_common(1)[0][0])

        # Tokenize answers (no special tokens — these are continuations)
        answer_tokens = model.tokenizer(
            best_answers, return_tensors="pt", padding=True,
            truncation=True, max_length=32, add_special_tokens=False,
        )

        nlls = compute_batch_answer_ppl(
            model,
            batch["pixel_values"],
            batch["prompt_ids"],
            batch["prompt_mask"],
            answer_tokens["input_ids"],
            answer_tokens["attention_mask"],
        )

        all_nlls.extend(nlls.cpu().tolist())
        done += B

        if done % (batch_size * 50) == 0 or done >= total:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            mean_so_far = sum(all_nlls) / len(all_nlls)
            print(f"  [{done}/{total}] {speed:.1f} samples/sec, mean NLL: {mean_so_far:.4f}")

    metrics = {
        "mean_answer_nll": float(np.mean(all_nlls)),
        "median_answer_nll": float(np.median(all_nlls)),
        "std_answer_nll": float(np.std(all_nlls)),
        "num_samples": len(all_nlls),
        "elapsed_sec": time.time() - t0,
    }
    return metrics


@torch.no_grad()
def evaluate_caption_ppl(
    model: VLMForScaling,
    dataset: COCOCaptionDataset,
    batch_size: int = 8,
) -> dict:
    """
    PPL-based captioning evaluation.

    Computes NLL of first reference caption given image + caption prompt.
    Should correlate strongly with pretraining val loss since both are
    captioning tasks.
    """
    loader = build_eval_dataloader(
        dataset,
        batch_size=batch_size,
        pad_token_id=model.tokenizer.pad_token_id or 0,
    )

    all_nlls = []
    total = len(dataset)
    done = 0
    t0 = time.time()

    for batch in loader:
        B = batch["pixel_values"].shape[0]
        references_list = batch["meta"]["references"]  # list of list[str]

        # Use first reference caption for each sample
        first_refs = [refs[0] for refs in references_list]

        # Tokenize captions
        caption_tokens = model.tokenizer(
            first_refs, return_tensors="pt", padding=True,
            truncation=True, max_length=128, add_special_tokens=False,
        )

        nlls = compute_batch_answer_ppl(
            model,
            batch["pixel_values"],
            batch["prompt_ids"],
            batch["prompt_mask"],
            caption_tokens["input_ids"],
            caption_tokens["attention_mask"],
        )

        all_nlls.extend(nlls.cpu().tolist())
        done += B

        if done % (batch_size * 50) == 0 or done >= total:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            mean_so_far = sum(all_nlls) / len(all_nlls)
            print(f"  [{done}/{total}] {speed:.1f} samples/sec, mean NLL: {mean_so_far:.4f}")

    metrics = {
        "mean_caption_nll": float(np.mean(all_nlls)),
        "median_caption_nll": float(np.median(all_nlls)),
        "std_caption_nll": float(np.std(all_nlls)),
        "num_samples": len(all_nlls),
        "elapsed_sec": time.time() - t0,
    }
    return metrics


def build_dataset(benchmark: str, benchmark_dir: str, model: VLMForScaling, max_samples: int | None):
    """Build the appropriate dataset for a benchmark."""
    bdir = Path(benchmark_dir)

    if benchmark == "vqav2":
        return VQAv2Dataset(
            questions_file=str(bdir / "vqav2" / "v2_OpenEnded_mscoco_val2014_questions.json"),
            annotations_file=str(bdir / "vqav2" / "v2_mscoco_val2014_annotations.json"),
            image_root=str(bdir / "coco" / "val2014"),
            image_processor=model.image_processor,
            tokenizer=model.tokenizer,
            max_samples=max_samples,
        )
    elif benchmark == "textvqa":
        return TextVQADataset(
            data_file=str(bdir / "textvqa" / "TextVQA_0.5.1_val.json"),
            image_root=str(bdir / "textvqa" / "train_images"),
            image_processor=model.image_processor,
            tokenizer=model.tokenizer,
            max_samples=max_samples,
        )
    elif benchmark == "coco_caption":
        return COCOCaptionDataset(
            karpathy_file=str(bdir / "coco_caption" / "dataset_coco.json"),
            image_root=str(bdir / "coco"),
            image_processor=model.image_processor,
            tokenizer=model.tokenizer,
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def evaluate_checkpoint(
    checkpoint_dir: str,
    benchmarks: list[str],
    benchmark_dir: str,
    batch_size: int,
    max_samples: int | None,
    output_dir: str | None,
    eval_mode: str = "generate",
):
    """Evaluate one checkpoint on specified benchmarks."""
    ckpt_name = Path(checkpoint_dir).name
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name} (mode={eval_mode})")
    print(f"{'='*60}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_checkpoint(checkpoint_dir, device=device)
    print(f"  Model loaded: {model.llm_name}, T={model.adapter.num_queries}, "
          f"N_A={model.adapter.num_params:,}")

    # Load training val loss
    result_path = Path(checkpoint_dir) / "result.json"
    train_val_loss = None
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)
        train_val_loss = result.get("best_val_loss")
        print(f"  Training val loss: {train_val_loss}")

    all_results = {
        "checkpoint": ckpt_name,
        "llm_name": model.llm_name,
        "num_queries": model.adapter.num_queries,
        "adapter_params": model.adapter.num_params,
        "train_val_loss": train_val_loss,
        "eval_mode": eval_mode,
    }

    for benchmark in benchmarks:
        print(f"\n--- {benchmark} ({eval_mode}) ---")
        dataset = build_dataset(benchmark, benchmark_dir, model, max_samples)
        print(f"  Dataset size: {len(dataset):,}")

        if eval_mode == "ppl":
            if benchmark in ("vqav2", "textvqa"):
                metrics = evaluate_vqa_ppl(model, dataset, batch_size=batch_size)
            elif benchmark == "coco_caption":
                metrics = evaluate_caption_ppl(model, dataset, batch_size=batch_size)
            else:
                continue
        else:  # generate
            if benchmark in ("vqav2", "textvqa"):
                metrics = evaluate_vqa(model, dataset, batch_size=batch_size)
            elif benchmark == "coco_caption":
                metrics = evaluate_caption(model, dataset, batch_size=batch_size)
            else:
                continue

        all_results[benchmark] = metrics
        print(f"  Results: {json.dumps(metrics, indent=2)}")

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result_file = out_path / f"eval_{ckpt_name}.json"
        with open(result_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved: {result_file}")

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="G10 Downstream Evaluation")

    # Input
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to single checkpoint directory")
    parser.add_argument("--batch_config", type=str, default=None,
                        help="Text file listing checkpoint directories (one per line)")

    # Benchmark
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["vqav2", "textvqa", "coco_caption", "all"],
                        help="Which benchmark(s) to evaluate")
    parser.add_argument("--benchmark_dir", type=str, default="data/benchmarks",
                        help="Root directory for benchmark data")

    # Inference
    parser.add_argument("--eval_mode", type=str, default="generate",
                        choices=["generate", "ppl"],
                        help="generate: text gen + accuracy; ppl: perplexity of correct answer")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit eval samples (for quick testing)")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save evaluation results")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.benchmark == "all":
        benchmarks = ["vqav2", "textvqa", "coco_caption"]
    else:
        benchmarks = [args.benchmark]

    # Collect checkpoint list
    checkpoints = []
    if args.checkpoint:
        checkpoints.append(args.checkpoint)
    elif args.batch_config:
        with open(args.batch_config) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    checkpoints.append(line)
    else:
        print("Error: provide --checkpoint or --batch_config")
        sys.exit(1)

    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Benchmark data: {args.benchmark_dir}")

    all_results = []
    for ckpt in checkpoints:
        result = evaluate_checkpoint(
            checkpoint_dir=ckpt,
            benchmarks=benchmarks,
            benchmark_dir=args.benchmark_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            eval_mode=args.eval_mode,
        )
        all_results.append(result)

    # Save combined summary
    if args.output_dir and len(all_results) > 1:
        suffix = "_ppl" if args.eval_mode == "ppl" else ""
        summary_path = Path(args.output_dir) / f"eval_summary{suffix}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY (mode={args.eval_mode})")
    print(f"{'='*80}")

    if args.eval_mode == "ppl":
        header = f"{'Checkpoint':<40} {'ValLoss':>8}"
        for b in benchmarks:
            if b in ("vqav2", "textvqa"):
                header += f" {b+' NLL':>12}"
            elif b == "coco_caption":
                header += f" {'Cap NLL':>12}"
        print(header)
        print("-" * 80)

        for r in all_results:
            vl = r.get('train_val_loss')
            row = f"{r['checkpoint']:<40} {vl if vl is not None else 'N/A':>8}"
            for b in benchmarks:
                if b in r:
                    if b in ("vqav2", "textvqa"):
                        row += f" {r[b].get('mean_answer_nll', 0):>12.4f}"
                    elif b == "coco_caption":
                        row += f" {r[b].get('mean_caption_nll', 0):>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
    else:
        header = f"{'Checkpoint':<40} {'ValLoss':>8}"
        for b in benchmarks:
            if b in ("vqav2", "textvqa"):
                header += f" {b+' Acc':>12}"
            elif b == "coco_caption":
                header += f" {'CIDEr':>12}"
        print(header)
        print("-" * 80)

        for r in all_results:
            vl = r.get('train_val_loss')
            row = f"{r['checkpoint']:<40} {vl if vl is not None else 'N/A':>8}"
            for b in benchmarks:
                if b in r:
                    if b in ("vqav2", "textvqa"):
                        row += f" {r[b].get('vqa_accuracy', 0):>11.2f}%"
                    elif b == "coco_caption":
                        cider = r[b].get("CIDEr", r[b].get("CIDEr_approx", 0))
                        row += f" {cider:>11.2f}%"
                else:
                    row += f" {'N/A':>12}"
            print(row)


if __name__ == "__main__":
    main()

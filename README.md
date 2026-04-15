# What Matters for Vision Adapters? Data and Capacity Trump Token Count

Code for the paper *"Scaling Laws for Vision-Language Model Adapters: Data and Capacity Trump Token Count"*.

We derive the first joint scaling law $L(N_L, D, T, N_A)$ for vision-language adapter design, covering LLM scale, data size, visual token count, and adapter capacity. We show that a fixed suboptimal learning rate qualitatively reverses scaling conclusions, and that at the proper learning rate an optimal adapter capacity exists ("bigger is not always better").

## Setup

```bash
# Create conda environment
conda create -n vlm python=3.10 -y
conda activate vlm
pip install -r requirements.txt
```

### Data

Download LLaVA-Pretrain-558K:
```bash
python scripts/download_data.py
```

Download base models (SigLIP-SO400M/14-384, Qwen2.5-{1.5B,3B,7B,14B}):
```bash
python scripts/download_models.py
```

## Training

Train a single configuration:
```bash
python train.py --config configs/g26/g26_14B_T64_M_d552k_s42_lr3e4.yaml
```

Run an experiment group:
```bash
source setup_env.sh
bash scripts/run.sh g26
```

All 441 YAML configs are in `configs/`. Each specifies the LLM, adapter level, token count, data size, learning rate, and seed.

### Key hyperparameters
- Optimizer: AdamW ($\beta_1$=0.9, $\beta_2$=0.95, weight decay 0.1)
- LR schedule: cosine with 2% warmup
- Batch size: 32 (effective; micro-batch × gradient accumulation)
- Precision: bfloat16
- Only the adapter is trained; vision encoder and LLM are frozen

## Evaluation

### Downstream (VQAv2, TextVQA, COCO Caption)

```bash
python eval.py --config configs/g26/g26_14B_T64_M_d552k_s42_lr3e4.yaml \
               --checkpoint checkpoints/g26_14B_T64_M_d552k_s42_lr3e4/adapter_best.pt \
               --eval_mode ppl
```

### Scaling law fit

Reproduce the joint scaling law from paper Section 5:
```bash
python analysis/scaling_fit.py \
    --csv analysis/results_full.csv \
    --fit_groups g26 g27 g28 g29 \
    --exclude_size 0.5B \
    --leave_one_out \
    --bootstrap 2000 \
    --output_dir analysis/results_paper
```

### Iso-FLOP analysis

```bash
python analysis/iso_flop.py
```

### Latency benchmark

```bash
python analysis/latency_benchmark.py --llms 3B 14B --tokens 32 64 128 256
```

### Gradient norm analysis

```bash
python analysis/measure_grad_norms.py
```

## Key results

| Metric | Value |
|--------|-------|
| Joint law $R^2$ | 0.847 |
| CV MAPE (5-fold) | 3.25% $\pm$ 0.77 |
| 32B extrapolation MAPE | 14.5% |
| Downstream correlation (mean \|r\|) | 0.73 |

Variable importance hierarchy: $D \gg N_A \gg N_L \gg T$

## Repository structure

```
├── train.py                  # Training entry point
├── eval.py                   # Downstream evaluation
├── src/
│   ├── model.py              # VLMForScaling (SigLIP + Adapter + Qwen2.5)
│   ├── perceiver_resampler.py # Perceiver Resampler adapter
│   ├── adapters.py           # MLP Projector + Q-Former adapter
│   ├── data.py               # Dataset and data loading
│   ├── trainer.py            # Training loop
│   ├── eval_datasets.py      # Benchmark dataset loaders
│   └── metrics.py            # VQA accuracy, CIDEr
├── analysis/
│   ├── scaling_fit.py        # Joint law fitting + diagnostics
│   ├── iso_flop.py           # Compute-optimal analysis
│   ├── latency_benchmark.py  # Inference latency measurement
│   ├── collect_all.py        # Result collection
│   └── plot_figures.py       # Paper figure generation
├── scripts/
│   ├── run.sh                # Multi-GPU experiment runner
│   ├── setup.sh              # Environment setup
│   └── generate_configs.py   # Config generation
├── configs/                  # 437 YAML experiment configurations
│   ├── g0/ ... g9/           # Phase 1 (LR=1e-4)
│   ├── g26/ ... g29/         # Phase 2 (LR=3e-4, primary fit)
│   ├── g30/ ... g33/         # 32B experiments
│   └── g34/                  # Q-Former capacity sweep
└── requirements.txt
```

## Hardware

- Phase 1: 8x NVIDIA A6000 (48 GB)
- Phase 2: 2x A6000 + 8x RTX 5090 (32 GB)
- Total compute: ~800 A6000-equivalent GPU-hours

## Base models

| Component | Model ID |
|-----------|----------|
| Vision encoder | `google/siglip-so400m-patch14-384` |
| LLM (1.5B) | `Qwen/Qwen2.5-1.5B` |
| LLM (3B) | `Qwen/Qwen2.5-3B` |
| LLM (7B) | `Qwen/Qwen2.5-7B` |
| LLM (14B) | `Qwen/Qwen2.5-14B` |
| LLM (32B) | `Qwen/Qwen2.5-32B` |

## License

MIT

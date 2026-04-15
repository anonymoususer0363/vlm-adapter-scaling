#!/bin/bash
# 범용 서버 세팅 스크립트
# 어떤 서버든 vlmadapter/ 폴더 옮긴 후 이 스크립트 한 번 실행
#
# Usage:
#   cd vlmadapter
#   bash scripts/setup.sh
#
# 자동 처리:
# 1. conda 환경 생성 + 패키지 설치
# 2. hf_cache/ 자동 감지 (모델 재다운 불필요)
# 3. 데이터 확인
# 4. Config 생성 (경로 자동 설정)
# 5. Smoke test
# 6. 서버 정보 출력 (GPU, VRAM, 디스크)

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

# Load environment variables (NAS paths, HF_HOME, etc.)
source "${PROJECT_DIR}/setup_env.sh"

echo "============================================"
echo "VLM Adapter Scaling Law — Server Setup"
echo "============================================"
echo "Project dir: ${PROJECT_DIR}"
echo ""

# ─────────────── Step 0: 서버 정보 ───────────────
echo "=== Server Info ==="
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs)
    echo "  GPU: ${GPU_COUNT}× ${GPU_NAME} (${GPU_VRAM})"
else
    echo "  GPU: nvidia-smi not found"
fi

echo "  Disk: $(df -h "${PROJECT_DIR}" | tail -1 | awk '{print $4 " free on " $6}')"
echo ""

# ─────────────── Step 1: 환경 세팅 ───────────────
echo "=== Step 1: Environment Setup ==="

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"

    if conda info --envs 2>/dev/null | grep -q "vlm "; then
        echo "conda env 'vlm' already exists."
        conda activate vlm
    else
        echo "Creating conda env 'vlm'..."
        conda create -n vlm python=3.11 -y
        conda activate vlm
    fi
else
    echo "conda not found, using system python: $(python3 --version 2>&1)"
fi

echo "Installing requirements..."
pip install -r requirements.txt -q 2>&1 | tail -3
echo "Done."

# ─────────────── Step 2: HF_HOME 설정 ───────────────
echo ""
echo "=== Step 2: Model Cache ==="

# HF_HOME already set by setup_env.sh; check for cached models
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -d "${HF_HUB}" ]; then
    echo "HF_HOME: ${HF_HOME:-$HOME/.cache/huggingface}"
    echo "Checking cached models:"
    for m in google--siglip-so400m-patch14-384 google--siglip-so400m-patch14-224 \
             Qwen--Qwen2.5-0.5B Qwen--Qwen2.5-1.5B Qwen--Qwen2.5-3B \
             Qwen--Qwen2.5-7B Qwen--Qwen2.5-14B Qwen--Qwen2.5-32B; do
        d="${HF_HUB}/models--${m}"
        if [ -d "$d" ]; then
            echo "  ✓ ${m}  ($(du -sh $d | cut -f1))"
        else
            echo "  ✗ ${m} — will download on first use"
        fi
    done
else
    echo "No model cache found."
    echo "Models will download to ~/.cache/huggingface/ on first use."
    echo ""
    echo "Tip: python scripts/download_models.py 0.5B 1.5B 3B 7B 14B 32B"
fi

# ─────────────── Step 3: 데이터 확인 ───────────────
echo ""
echo "=== Step 3: Data Check ==="

DATA_DIR="${VLM_DATA_DIR}"
echo "  Data dir: ${DATA_DIR}"

# 이미지 디렉토리 자동 감지 (images/ 또는 숫자 폴더)
IMAGE_ROOT=""
if [ -d "${DATA_DIR}/llava_pretrain" ]; then
    # LLaVA-Pretrain은 00000/ 00001/ 형태로 풀림
    SUBDIRS=$(find "${DATA_DIR}/llava_pretrain" -maxdepth 1 -type d -name "[0-9]*" | head -1)
    if [ -n "$SUBDIRS" ]; then
        IMAGE_ROOT="${DATA_DIR}/llava_pretrain"
        IMG_COUNT=$(find "${DATA_DIR}/llava_pretrain" -maxdepth 2 -type f -name "*.jpg" | head -1000 | wc -l)
        echo "  ✓ images: ${IMG_COUNT}+ files in ${DATA_DIR}/llava_pretrain/"
    fi
fi

if [ -z "$IMAGE_ROOT" ] && [ -d "${DATA_DIR}/images" ]; then
    IMAGE_ROOT="${DATA_DIR}/images"
    echo "  ✓ images: ${DATA_DIR}/images/"
fi

if [ -z "$IMAGE_ROOT" ]; then
    echo "  ✗ No image directory found"
    echo "  Run: python scripts/download_data.py --phase 1 --data_dir ${DATA_DIR}"
fi

if [ -f "${DATA_DIR}/processed/train.jsonl" ]; then
    TRAIN_LINES=$(wc -l < "${DATA_DIR}/processed/train.jsonl")
    VAL_LINES=$(wc -l < "${DATA_DIR}/processed/val.jsonl" 2>/dev/null || echo "0")
    echo "  ✓ train.jsonl: ${TRAIN_LINES} lines"
    echo "  ✓ val.jsonl:   ${VAL_LINES} lines"
else
    echo "  ✗ ${DATA_DIR}/processed/ not found"
    if [ -f "${DATA_DIR}/llava_pretrain/blip_laion_cc_sbu_558k.json" ]; then
        echo "  Converting to JSONL..."
        python scripts/download_data.py --phase 2 --data_dir "${DATA_DIR}"
    else
        echo "  Run: python scripts/download_data.py --phase 1 --data_dir ${DATA_DIR}"
        echo "       python scripts/download_data.py --phase 2 --data_dir ${DATA_DIR}"
    fi
fi

# ─────────────── Step 4: Config 생성 ───────────────
echo ""
echo "=== Step 4: Generate Configs ==="

# image_root 자동 감지된 값 사용
IMAGE_ROOT=${IMAGE_ROOT:-"${DATA_DIR}/llava_pretrain"}
CKPT_DIR="${VLM_CHECKPOINT_DIR}"

rm -rf configs/g*
python scripts/generate_configs.py \
    --train_data "${DATA_DIR}/processed/train.jsonl" \
    --val_data "${DATA_DIR}/processed/val.jsonl" \
    --image_root "${IMAGE_ROOT}" \
    --output_dir "${CKPT_DIR}"

# ─────────────── Step 5: Smoke Test ───────────────
echo ""
echo "=== Step 5: Smoke Test ==="
python scripts/smoke_test.py

# ─────────────── Step 6: 환경변수 영속 ───────────────
echo ""
echo "=== Step 6: Persist Environment ==="

# Add 'source setup_env.sh' to bashrc for persistent env
if ! grep -q "vlmadapter/setup_env.sh" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# VLM Adapter project" >> ~/.bashrc
    echo "source ${PROJECT_DIR}/setup_env.sh" >> ~/.bashrc
    echo "Added setup_env.sh to ~/.bashrc"
else
    echo "setup_env.sh already in ~/.bashrc"
fi

# ─────────────── Summary ───────────────
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "실행:"
echo "  conda activate vlm"
echo "  source setup_env.sh"
echo "  bash scripts/run.sh all                  # 전체 실행"
echo "  bash scripts/run.sh g0 g1 g2             # 우선 그룹"
echo "  bash scripts/run.sh all --dry-run         # 계획만"
echo "  RESERVE_GPUS=2 bash scripts/run.sh all   # GPU 남겨두기"
echo ""
echo "모니터링:"
echo "  bash scripts/gpu_status.sh"
echo "  watch -n 10 bash scripts/gpu_status.sh"

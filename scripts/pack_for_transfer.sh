#!/bin/bash
# 모든 모델 + 데이터를 프로젝트 폴더 안으로 모아서
# vlmadapter/ 하나만 옮기면 다른 서버에서 바로 실행 가능하게 만듦
#
# Usage: bash scripts/pack_for_transfer.sh
#
# 결과:
#   vlmadapter/
#     hf_cache/hub/models--...   ← HF 모델 캐시 (symlink resolved copy)
#     data/processed/            ← JSONL train/val
#     data/llava_pretrain/       ← 이미지 (00000/, 00001/ ...)

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

# HF cache source: prefer VLM_HF_HOME > HF_HOME > default
HF_CACHE_SRC="${VLM_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
HF_CACHE_DST="${PROJECT_DIR}/hf_cache"

echo "============================================"
echo "Pack for Transfer"
echo "============================================"
echo "  HF cache source: ${HF_CACHE_SRC}"
echo ""

# ─────────────── Check prerequisites ───────────────
echo "=== Checking downloads ==="

REQUIRED_MODELS=(
    "models--google--siglip-so400m-patch14-384"
    "models--google--siglip-so400m-patch14-224"
    "models--Qwen--Qwen2.5-0.5B"
    "models--Qwen--Qwen2.5-1.5B"
    "models--Qwen--Qwen2.5-3B"
    "models--Qwen--Qwen2.5-7B"
    "models--Qwen--Qwen2.5-14B"
    "models--Qwen--Qwen2.5-32B"
)

MISSING=0
for model in "${REQUIRED_MODELS[@]}"; do
    src="${HF_CACHE_SRC}/hub/${model}"
    if [ -d "$src" ]; then
        incomplete=$(find "$src" -name "*.incomplete" 2>/dev/null | wc -l)
        if [ "$incomplete" -gt 0 ]; then
            echo "  ⏳ ${model} — still downloading (${incomplete} incomplete)"
            MISSING=$((MISSING + 1))
        else
            size=$(du -sh "$src" 2>/dev/null | cut -f1)
            echo "  ✓  ${model} (${size})"
        fi
    else
        echo "  ✗  ${model} — NOT FOUND"
        MISSING=$((MISSING + 1))
    fi
done

echo ""

# Check data — LLaVA-Pretrain uses numbered folders (00000/, 00001/ ...) not images/
DATA_DIR="${PROJECT_DIR}/data"
if [ -d "${DATA_DIR}/llava_pretrain" ]; then
    SUBDIRS=$(find "${DATA_DIR}/llava_pretrain" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | head -1)
    if [ -n "$SUBDIRS" ]; then
        IMG_COUNT=$(find "${DATA_DIR}/llava_pretrain" -maxdepth 2 -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | head -1000 | wc -l)
        echo "  ✓  images (${IMG_COUNT}+ files in numbered folders)"
    else
        echo "  ✗  images — no numbered folders found in data/llava_pretrain/"
        MISSING=$((MISSING + 1))
    fi
else
    echo "  ✗  data/llava_pretrain/ — not found"
    MISSING=$((MISSING + 1))
fi

if [ -f "${DATA_DIR}/processed/train.jsonl" ]; then
    echo "  ✓  data/processed/train.jsonl"
else
    echo "  ✗  data/processed/train.jsonl — run: python scripts/download_data.py --phase 2 --data_dir data"
    MISSING=$((MISSING + 1))
fi

if [ ${MISSING} -gt 0 ]; then
    echo ""
    echo "ERROR: ${MISSING} items not ready. 다운로드 완료 후 다시 실행하세요."
    exit 1
fi

# ─────────────── Copy HF models ───────────────
echo ""
echo "=== Copying model cache into project ==="

mkdir -p "${HF_CACHE_DST}/hub"

for model in "${REQUIRED_MODELS[@]}"; do
    src="${HF_CACHE_SRC}/hub/${model}"
    dst="${HF_CACHE_DST}/hub/${model}"

    if [ -d "$dst" ]; then
        echo "  SKIP ${model} (already packed)"
        continue
    fi

    echo "  Copying ${model}..."
    # cp -rL resolves symlinks (HF cache uses symlinks to blobs)
    cp -rL "$src" "$dst"
    echo "    $(du -sh "$dst" | cut -f1)"
done

echo ""
echo "=== Summary ==="
TOTAL_SIZE=$(du -sh "${PROJECT_DIR}" --exclude='.git' --exclude='__pycache__' 2>/dev/null | cut -f1)
echo "Total project size: ${TOTAL_SIZE}"
echo ""
echo "이제 vlmadapter/ 폴더 하나만 옮기면 됩니다."
echo ""
echo "다른 서버에서:"
echo "  cd vlmadapter"
echo "  bash scripts/setup.sh    # 자동 세팅 (conda + config + smoke test)"
echo "  bash scripts/run.sh all  # 실험 실행"

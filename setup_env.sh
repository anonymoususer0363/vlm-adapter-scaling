#!/bin/bash
# 프로젝트 환경 변수 설정 — 모든 스크립트/실행 전에 source 하세요
#
# Usage:
#   source setup_env.sh                    # 기본 (모든 것 프로젝트 안)
#
# NAS 사용 시:
#   export VLM_NAS=/mnt/148TB/hyeongjin
#   source setup_env.sh
#   → HF_HOME=$VLM_NAS/hf_cache, data=$VLM_NAS/data, checkpoints=$VLM_NAS/checkpoints

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ─── NAS / 외부 스토리지 설정 ───
# VLM_NAS가 설정되어 있으면 데이터/모델/체크포인트를 NAS 경로 사용
if [ -n "${VLM_NAS}" ]; then
    export VLM_HF_HOME="${VLM_HF_HOME:-${VLM_NAS}/hf_cache}"
    export VLM_DATA_DIR="${VLM_DATA_DIR:-${VLM_NAS}/data}"
    export VLM_CHECKPOINT_DIR="${VLM_CHECKPOINT_DIR:-${VLM_NAS}/checkpoints}"
fi

# ─── HF 모델 캐시 ───
# 우선순위: VLM_HF_HOME > 프로젝트 내 hf_cache/ > 기본 ~/.cache
if [ -n "${VLM_HF_HOME}" ] && [ -d "${VLM_HF_HOME}" ]; then
    export HF_HOME="${VLM_HF_HOME}"
    export TRANSFORMERS_CACHE="${VLM_HF_HOME}/hub"
elif [ -d "${PROJECT_DIR}/hf_cache/hub" ]; then
    export HF_HOME="${PROJECT_DIR}/hf_cache"
    export TRANSFORMERS_CACHE="${PROJECT_DIR}/hf_cache/hub"
fi

# ─── 데이터/체크포인트 경로 (기본값: 프로젝트 안) ───
export VLM_DATA_DIR="${VLM_DATA_DIR:-${PROJECT_DIR}/data}"
export VLM_CHECKPOINT_DIR="${VLM_CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"

# ─── WandB ───
export WANDB_PROJECT="vlm-adapter-scaling"

# 출력 (QUIET=1이면 생략 — run.sh 등에서 사용)
if [ "${QUIET:-0}" != "1" ]; then
    echo "Environment ready."
    echo "  Project:     ${PROJECT_DIR}"
    [ -n "${HF_HOME}" ] && echo "  HF_HOME:     ${HF_HOME}"
    echo "  Data:        ${VLM_DATA_DIR}"
    echo "  Checkpoints: ${VLM_CHECKPOINT_DIR}"
fi

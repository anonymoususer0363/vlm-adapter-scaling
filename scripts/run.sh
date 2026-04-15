#!/bin/bash
# 서버 무관 실험 실행 스크립트
# nvidia-smi로 GPU VRAM/점유 자동 감지 → 빈 GPU에 자동 할당
#
# A6000×2, A6000×8, 5090×8, B200×64 등 어떤 서버든 동작
#
# Usage:
#   bash scripts/run.sh all                       # 전체 실행
#   bash scripts/run.sh g0 g1 g2                  # 특정 그룹만
#   bash scripts/run.sh all --dry-run              # 실행 안하고 계획만
#   RESERVE_GPUS=2 bash scripts/run.sh all        # GPU 2장 남겨두기
#   POLL_INTERVAL=60 bash scripts/run.sh all      # 1분마다 GPU 체크

set -e

# ─────────────── Config ───────────────
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# Load environment (HF_HOME, VLM_DATA_DIR, VLM_CHECKPOINT_DIR, etc.)
QUIET=1 source "${PROJECT_DIR}/setup_env.sh"

# Resolve python path once (supports nohup/non-interactive shells)
export PYTHON="${PYTHON:-$(which python)}"

FREE_THRESHOLD=${FREE_THRESHOLD:-100}   # MB 이하면 빈 GPU
POLL_INTERVAL=${POLL_INTERVAL:-30}      # 초
RESERVE_GPUS=${RESERVE_GPUS:-0}         # 남겨둘 GPU 수
LOG_DIR="${PROJECT_DIR}/logs"
DRY_RUN=false
SKIP_DONE=true

mkdir -p "${LOG_DIR}"

# ─────────────── Parse args ───────────────
RUN_GROUPS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --no-skip) SKIP_DONE=false ;;
        *) RUN_GROUPS+=("$arg") ;;
    esac
done

if [ ${#RUN_GROUPS[@]} -eq 0 ]; then
    echo "Usage: bash scripts/run.sh <group|all> [--dry-run] [--no-skip]"
    echo ""
    echo "Options:"
    echo "  --dry-run     실행 안하고 계획만 출력"
    echo "  --no-skip     이미 완료된 실험도 재실행"
    echo ""
    echo "Environment:"
    echo "  RESERVE_GPUS=N    항상 N장 남겨두기 (동료 배려)"
    echo "  POLL_INTERVAL=30  빈 GPU 체크 간격 (초)"
    echo "  FREE_THRESHOLD=100  이 MB 이하면 빈 GPU"
    exit 1
fi

if [ "${RUN_GROUPS[0]}" == "all" ]; then
    RUN_GROUPS=(g0 g1 g2 g3 g7 g4 g5 g8 g9 g6)
fi

# ─────────────── GPU Detection ───────────────

detect_server() {
    # 서버 정보 자동 감지 (CUDA_VISIBLE_DEVICES 존중)
    local gpu_count
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        gpu_count=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    else
        gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    fi
    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    local vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    local vram_gb=$((vram_mb / 1024))

    # Human-readable info to stderr, data to stdout
    echo "  GPUs: ${gpu_count}× ${gpu_name} (${vram_gb}GB each)" >&2
    echo "${gpu_count} ${vram_gb}"
}

get_free_gpu_list() {
    # 실제 빈 GPU 번호 목록 반환 (메모리 사용량 기준, CUDA_VISIBLE_DEVICES 존중)
    local allowed_gpus=""
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        allowed_gpus=",${CUDA_VISIBLE_DEVICES},"
    fi
    local free=()
    while IFS=',' read -r idx mem_used; do
        idx=$(echo "$idx" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ' | tr -d 'MiB')
        # Skip GPUs not in CUDA_VISIBLE_DEVICES
        if [ -n "$allowed_gpus" ] && [[ "$allowed_gpus" != *",${idx},"* ]]; then
            continue
        fi
        if [ -n "$idx" ] && [ "$mem_used" -le "$FREE_THRESHOLD" ] 2>/dev/null; then
            free+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "${free[@]}"
}

get_gpus_needed() {
    # LLM 크기 + GPU VRAM → 필요 GPU 수 자동 계산
    local config=$1
    local vram_gb=$2
    local llm_name=$(grep "llm_name" "$config" | head -1 | awk '{print $2}')

    # LLM bf16 weight size (GB)
    local llm_gb=0
    case "$llm_name" in
        *0.5B*) llm_gb=1 ;;
        *1.5B*) llm_gb=3 ;;
        *3B*)   llm_gb=6 ;;
        *7B*)   llm_gb=14 ;;
        *14B*)  llm_gb=28 ;;
        *32B*)  llm_gb=64 ;;
        *)      llm_gb=10 ;;
    esac

    # Total: LLM + VE(~1GB) + adapter(<1GB) + activations
    local overhead=6
    if [ "$llm_gb" -ge 64 ]; then
        # 32B GPU requirement depends on VRAM:
        #   48GB (A6000): 2 GPU, 32GB (5090): 4 GPU
        if [ "$vram_gb" -ge 40 ]; then
            overhead=32   # ceil((64+32)/48)=2
        else
            overhead=60   # ceil((64+60)/31)=4
        fi
    fi
    local total_need=$((llm_gb + overhead))

    # ceil(total_need / vram_gb)
    local gpus=$(( (total_need + vram_gb - 1) / vram_gb ))
    [ "$gpus" -lt 1 ] && gpus=1

    echo "$gpus"
}

check_done() {
    local config=$1
    local run_name=$(grep "run_name" "$config" | head -1 | awk '{print $2}')
    local out_dir=$(grep "output_dir" "$config" | head -1 | awk '{print $2}')
    out_dir="${out_dir:-${VLM_CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}}"
    [ -f "${out_dir}/${run_name}/result.json" ]
}

# ─────────────── Run ───────────────

run_one() {
    local config=$1
    local gpu_ids=$2
    local run_name=$(basename "$config" .yaml)

    echo "[$(date '+%m-%d %H:%M:%S')] START ${run_name}  GPUs=[${gpu_ids}]"

    CUDA_VISIBLE_DEVICES=${gpu_ids} "$PYTHON" "${PROJECT_DIR}/train.py" \
        --config "${config}" \
        --use_wandb \
        2>&1 | tee "${LOG_DIR}/${run_name}.log"

    local st=$?
    if [ $st -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M:%S')] DONE ${run_name}"
    else
        echo "[$(date '+%m-%d %H:%M:%S')] FAIL ${run_name} (exit ${st})" | tee -a "${LOG_DIR}/failures.log"
    fi
}

# ─────────────── Main ───────────────

echo "============================================"
echo "VLM Adapter Scaling — Universal GPU Runner"
echo "============================================"

# Detect server
read GPU_COUNT VRAM_GB <<< $(detect_server)
echo "  Reserve: ${RESERVE_GPUS} GPUs"
echo "  Groups:  ${RUN_GROUPS[*]}"
echo ""

# Collect pending configs
declare -a CONFIGS
declare -a GPU_REQS

for group in "${RUN_GROUPS[@]}"; do
    config_dir="${PROJECT_DIR}/configs/${group}"
    [ ! -d "$config_dir" ] && continue

    for config in $(ls "${config_dir}"/*.yaml 2>/dev/null | sort); do
        if [ "$SKIP_DONE" = true ] && check_done "$config"; then
            continue
        fi

        needed=$(get_gpus_needed "$config" "$VRAM_GB")

        # Skip if impossible (needs more GPUs than available)
        if [ "$needed" -gt "$GPU_COUNT" ]; then
            echo "  SKIP $(basename $config): needs ${needed} GPUs, have ${GPU_COUNT}"
            continue
        fi

        CONFIGS+=("$config")
        GPU_REQS+=("$needed")
    done
done

TOTAL=${#CONFIGS[@]}
echo "Pending: ${TOTAL} experiments"

# Summary by GPU requirement
for req in 1 2 4 8 16; do
    cnt=0
    for r in "${GPU_REQS[@]}"; do [ "$r" -eq "$req" ] && cnt=$((cnt + 1)); done
    [ $cnt -gt 0 ] && echo "  ${req}-GPU: ${cnt} runs"
done
echo ""

[ $TOTAL -eq 0 ] && echo "Nothing to run!" && exit 0

if [ "$DRY_RUN" = true ]; then
    echo "=== Dry Run ==="
    for ((i=0; i<TOTAL; i++)); do
        echo "  [${GPU_REQS[$i]} GPU] $(basename ${CONFIGS[$i]})"
    done
    exit 0
fi

# ─────────────── Execution loop ───────────────
COMPLETED=0
RUNNING_PIDS=()
RUNNING_GPUS=()
NEXT=0

while [ $COMPLETED -lt $TOTAL ]; do
    # Clean up finished
    NPIDS=() NGPUS=()
    for ((j=0; j<${#RUNNING_PIDS[@]}; j++)); do
        if kill -0 "${RUNNING_PIDS[$j]}" 2>/dev/null; then
            NPIDS+=("${RUNNING_PIDS[$j]}")
            NGPUS+=("${RUNNING_GPUS[$j]}")
        else
            wait "${RUNNING_PIDS[$j]}" 2>/dev/null
            COMPLETED=$((COMPLETED + 1))
            echo "[Progress] ${COMPLETED}/${TOTAL} done  (freed [${RUNNING_GPUS[$j]}])"
        fi
    done
    RUNNING_PIDS=("${NPIDS[@]}")
    RUNNING_GPUS=("${NGPUS[@]}")

    # Try to launch new jobs
    LAUNCHED=false
    while [ $NEXT -lt $TOTAL ]; do
        need="${GPU_REQS[$NEXT]}"

        # Get currently free GPUs
        read -ra FREE <<< "$(get_free_gpu_list)"

        # Exclude GPUs we're already using
        AVAIL=()
        for gpu in "${FREE[@]}"; do
            used=false
            for ug in "${RUNNING_GPUS[@]}"; do
                IFS=',' read -ra uarr <<< "$ug"
                for u in "${uarr[@]}"; do
                    [ "$gpu" == "$u" ] && used=true
                done
            done
            [ "$used" = false ] && AVAIL+=("$gpu")
        done

        # Check if enough free (including reserve)
        if [ ${#AVAIL[@]} -lt $((need + RESERVE_GPUS)) ]; then
            break
        fi

        # Allocate
        ALLOC=()
        for ((k=0; k<need; k++)); do ALLOC+=("${AVAIL[$k]}"); done
        GPU_STR=$(IFS=,; echo "${ALLOC[*]}")

        run_one "${CONFIGS[$NEXT]}" "$GPU_STR" &
        RUNNING_PIDS+=($!)
        RUNNING_GPUS+=("$GPU_STR")
        NEXT=$((NEXT + 1))
        LAUNCHED=true
        sleep 3
    done

    # Wait if nothing launched
    if [ "$LAUNCHED" = false ] && [ $NEXT -lt $TOTAL ]; then
        echo "[$(date '+%H:%M:%S')] Waiting... (running: ${#RUNNING_PIDS[@]}, free: ${#AVAIL[@]}, need: ${need}, reserve: ${RESERVE_GPUS})"
        sleep ${POLL_INTERVAL}
    fi

    # All submitted, wait for remaining
    if [ $NEXT -ge $TOTAL ] && [ ${#RUNNING_PIDS[@]} -gt 0 ]; then
        sleep ${POLL_INTERVAL}
    fi
done

echo ""
echo "============================================"
echo "ALL ${TOTAL} EXPERIMENTS COMPLETE"
echo "============================================"

if [ -f "${LOG_DIR}/failures.log" ]; then
    FC=$(wc -l < "${LOG_DIR}/failures.log")
    [ "$FC" -gt 0 ] && echo "FAILURES (${FC}):" && cat "${LOG_DIR}/failures.log"
fi

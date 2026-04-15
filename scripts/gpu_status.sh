#!/bin/bash
# GPU 사용 현황 한눈에 보기
# Usage: bash scripts/gpu_status.sh
#        watch -n 5 bash scripts/gpu_status.sh   # 5초마다 자동 갱신

echo "$(date '+%Y-%m-%d %H:%M:%S') — GPU Status"
echo "==========================================="

# 전체 GPU 상태
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader | while IFS=',' read -r idx name mem_used mem_total util temp; do
    idx=$(echo "$idx" | tr -d ' ')
    mem_used=$(echo "$mem_used" | tr -d ' ')
    mem_total=$(echo "$mem_total" | tr -d ' ')
    util=$(echo "$util" | tr -d ' ')
    temp=$(echo "$temp" | tr -d ' ')

    mem_num=$(echo "$mem_used" | tr -d 'MiB')
    if [ "$mem_num" -le 50 ] 2>/dev/null; then
        status="FREE"
    else
        status="BUSY"
    fi

    printf "  GPU %2s [%4s] %8s / %8s  util=%s  temp=%s\n" "$idx" "$status" "$mem_used" "$mem_total" "$util" "$temp"
done

echo ""
FREE_COUNT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 <= 50' | wc -l)
TOTAL_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "Summary: ${FREE_COUNT}/${TOTAL_COUNT} GPUs free"

# 내 프로세스 확인
echo ""
echo "=== My Running Experiments ==="
ps aux | grep "python.*train.py" | grep -v grep | awk '{print "  PID=" $2 " " $11 " " $12 " " $13}' || echo "  (none)"

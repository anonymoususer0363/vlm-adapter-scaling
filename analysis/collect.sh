#!/bin/bash
# Collect experiment results using only bash + basic tools (no python/jq needed)
# Outputs CSV to analysis/results_all.csv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT="$SCRIPT_DIR/results_all.csv"
DIRS=("${VLM_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}")

# CSV header
echo "run_name,source,group,llm_size,final_val_loss,best_val_loss,total_steps,seen_pairs,adapter_params,vision_T0,vision_T,vision_rho,vision_N_A,vision_d_model,vision_adapter_num_layers,vision_image_size,llm_name,adapter_level,num_queries,num_samples,num_epochs,seed,use_lora,batch_size,lr" > "$OUTPUT"

declare -A SEEN
TOTAL=0
INCOMPLETE=0
INCOMPLETE_LIST=""
declare -A GROUPS

extract_json_val() {
    local file="$1" key="$2"
    local val
    val=$(grep "\"$key\"" "$file" 2>/dev/null | head -1 | sed 's/.*: *//;s/,.*//;s/"//g;s/ *$//')
    echo "$val"
}

for DIR in "${DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then continue; fi
    SRC="NAS"
    if [[ "$DIR" == *data1* ]]; then SRC="local"; fi

    for RD in "$DIR"/*/; do
        [ -d "$RD" ] || continue
        NAME=$(basename "$RD")
        RESULT="$RD/result.json"
        CONFIG="$RD/config.json"

        if [ ! -f "$RESULT" ]; then
            INCOMPLETE=$((INCOMPLETE + 1))
            INCOMPLETE_LIST="$INCOMPLETE_LIST  [$SRC] $NAME\n"
            continue
        fi

        RUN_NAME=$(extract_json_val "$RESULT" "run_name")
        [ -z "$RUN_NAME" ] && RUN_NAME="$NAME"

        # Skip duplicates
        if [ -n "${SEEN[$RUN_NAME]}" ]; then continue; fi
        SEEN[$RUN_NAME]=1

        # Extract from result.json
        FINAL=$(extract_json_val "$RESULT" "final_val_loss")
        BEST=$(extract_json_val "$RESULT" "best_val_loss")
        STEPS=$(extract_json_val "$RESULT" "total_steps")
        PAIRS=$(extract_json_val "$RESULT" "seen_pairs")
        APARAMS=$(extract_json_val "$RESULT" "adapter_params")
        VT0=$(extract_json_val "$RESULT" "vision_T0")
        VT=$(extract_json_val "$RESULT" "vision_T")
        VRHO=$(extract_json_val "$RESULT" "vision_rho")
        VNA=$(extract_json_val "$RESULT" "vision_N_A")
        VDM=$(extract_json_val "$RESULT" "vision_d_model")
        VANL=$(extract_json_val "$RESULT" "vision_adapter_num_layers")
        VIS=$(extract_json_val "$RESULT" "vision_image_size")

        # Extract from config.json
        LLM_NAME="" ALEVEL="" NQ="" NS="" NE="1" SEED="42" LORA="False" BS="" LR=""
        if [ -f "$CONFIG" ]; then
            LLM_NAME=$(extract_json_val "$CONFIG" "llm_name")
            ALEVEL=$(extract_json_val "$CONFIG" "adapter_level")
            NQ=$(extract_json_val "$CONFIG" "num_queries")
            NS=$(extract_json_val "$CONFIG" "num_samples")
            NE=$(extract_json_val "$CONFIG" "num_epochs")
            SEED=$(extract_json_val "$CONFIG" "seed")
            LORA=$(extract_json_val "$CONFIG" "use_lora")
            BS=$(extract_json_val "$CONFIG" "batch_size")
            LR=$(extract_json_val "$CONFIG" "lr")
        fi

        # Parse group
        GROUP=""
        if [[ "$RUN_NAME" == rerun* ]]; then
            GROUP="rerun"
        elif [[ "$RUN_NAME" == g* ]]; then
            GROUP=$(echo "$RUN_NAME" | cut -d_ -f1)
        fi

        # Parse llm_size
        LLM_SIZE=""
        for SZ in 0.5B 1.5B 32B 14B 7B 3B; do
            if [[ "$LLM_NAME" == *"$SZ"* ]]; then
                LLM_SIZE="$SZ"
                break
            fi
        done

        # Track groups
        GROUPS[$GROUP]=$(( ${GROUPS[$GROUP]:-0} + 1 ))
        TOTAL=$((TOTAL + 1))

        echo "$RUN_NAME,$SRC,$GROUP,$LLM_SIZE,$FINAL,$BEST,$STEPS,$PAIRS,$APARAMS,$VT0,$VT,$VRHO,$VNA,$VDM,$VANL,$VIS,$LLM_NAME,$ALEVEL,$NQ,$NS,$NE,$SEED,$LORA,$BS,$LR" >> "$OUTPUT"
    done
done

echo ""
echo "=== Collection Summary ==="
echo "Total experiments with results: $TOTAL"
echo "Output: $OUTPUT"
echo ""
echo "--- Per-group counts ---"
for G in $(echo "${!GROUPS[@]}" | tr ' ' '\n' | sort); do
    echo "  $G: ${GROUPS[$G]}"
done
echo ""
echo "--- Incomplete (no result.json): $INCOMPLETE ---"
echo -e "$INCOMPLETE_LIST"
echo ""
echo "--- Sample (first 3 lines after header) ---"
head -4 "$OUTPUT"
echo ""
echo "--- Sample (last 3 lines) ---"
tail -3 "$OUTPUT"

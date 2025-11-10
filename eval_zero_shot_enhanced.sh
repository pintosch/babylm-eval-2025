#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
EVAL_DIR=${3:-"evaluation_data/full_eval"}
SKIP_EXISTING=${4:-true}  # <-- set to false to force re-run all tasks
OUTPUT_DIR="results"

if [[ -z "$MODEL_PATH" || -z "$BACKEND" ]]; then
    echo "Usage: $0 MODEL_PATH BACKEND [EVAL_DIR] [SKIP_EXISTING=true|false]"
    exit 1
fi

# Determine reading backend
if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo "Reading backend: $BACKEND_READ"

# Extract model name
MODEL_NAME=$(basename "$MODEL_PATH")

# Helper function: run only if output doesn't exist or skipping disabled
run_if_needed() {
    local task=$1
    local dataset_dir=$2
    local dataset_name
    dataset_name=$(basename "$dataset_dir")

    local output_path="${OUTPUT_DIR}/${MODEL_NAME}/main/zero_shot/${BACKEND}/${task}/${dataset_name}"

    if [[ "$SKIP_EXISTING" == "true" && -d "$output_path" && -n "$(ls -A "$output_path" 2>/dev/null)" ]]; then
        echo "✅ Skipping ${task} (${dataset_name}) — results already exist at ${output_path}"
    else
        echo "🚀 Running ${task} (${dataset_name})..."
        python -m evaluation_pipeline.sentence_zero_shot.run \
            --model_path_or_name "$MODEL_PATH" \
            --backend "$BACKEND" \
            --task "$task" \
            --data_path "$dataset_dir" \
            --save_predictions
    fi
}

# Zero-shot tasks
run_if_needed "blimp" "${EVAL_DIR}/blimp_filtered"
run_if_needed "blimp" "${EVAL_DIR}/supplement_filtered"
run_if_needed "ewok" "${EVAL_DIR}/ewok_filtered"
run_if_needed "entity_tracking" "${EVAL_DIR}/entity_tracking"
run_if_needed "wug_adj" "${EVAL_DIR}/wug_adj_nominalization"
run_if_needed "wug_past" "${EVAL_DIR}/wug_past_tense"
run_if_needed "comps" "${EVAL_DIR}/comps"

# This one takes too long...
if false; then
    echo "🟠 Skipping ___"
else    
    # ___
    echo ""
fi


# Reading task
READING_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}/main/zero_shot/${BACKEND_READ}/reading"
if [[ "$SKIP_EXISTING" == "true" && -d "$READING_OUTPUT_PATH" && -n "$(ls -A "$READING_OUTPUT_PATH" 2>/dev/null)" ]]; then
    echo "✅ Skipping reading task — results already exist at ${READING_OUTPUT_PATH}"
else
    echo "🚀 Running reading task..."
    python -m evaluation_pipeline.reading.run \
        --model_path_or_name "$MODEL_PATH" \
        --backend "$BACKEND_READ" \
        --data_path "${EVAL_DIR}/reading/reading_data.csv"
fi

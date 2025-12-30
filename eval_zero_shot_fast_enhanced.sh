#!/bin/bash
MODEL_PATH=$1
REVISION_NAME=$2
BACKEND=$3
MODEL_VARIANT=${4:-""}  # Optional model variant
EVAL_DIR=${5:-"evaluation_data/fast_eval"}
SKIP_EXISTING=${6:-true}  # <-- new flag: set to false to force re-run all tasks
OUTPUT_DIR="results"

if [[ -z "$MODEL_PATH" || -z "$REVISION_NAME" || -z "$BACKEND" ]]; then
    echo "Usage: $0 MODEL_PATH REVISION_NAME BACKEND [MODEL_VARIANT] [EVAL_DIR] [SKIP_EXISTING=true|false]"
    exit 1
fi

# Determine reading backend
if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")

# Helper function to check if result exists and run evaluation if not
run_if_needed() {
    local task=$1
    local dataset_dir=$2
    local dataset_name
    dataset_name=$(basename "$dataset_dir")

    local output_path="${OUTPUT_DIR}/${MODEL_NAME}/${REVISION_NAME}/zero_shot/${BACKEND}/${task}/${dataset_name}"

    if [[ "$SKIP_EXISTING" == "true" && -d "$output_path" && -n "$(ls -A "$output_path" 2>/dev/null)" ]]; then
        echo "✅ Skipping ${task} (${dataset_name}) — results already exist at ${output_path}"
    else
        echo "🚀 Running ${task} (${dataset_name})..."
        # Build optional model_variant parameter
        if [[ -n "$MODEL_VARIANT" ]]; then
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task "$task" \
                --data_path "$dataset_dir" \
                --save_predictions \
                --revision_name "$REVISION_NAME" \
                --model_variant "$MODEL_VARIANT"
        else
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task "$task" \
                --data_path "$dataset_dir" \
                --save_predictions \
                --revision_name "$REVISION_NAME"
        fi
    fi
}

# Zero-shot tasks
run_if_needed "blimp" "${EVAL_DIR}/blimp_fast"
run_if_needed "blimp" "${EVAL_DIR}/supplement_fast"
run_if_needed "ewok" "${EVAL_DIR}/ewok_fast"
run_if_needed "wug_adj" "${EVAL_DIR}/wug_adj_nominalization"
run_if_needed "wug_past" "${EVAL_DIR}/wug_past_tense"
# This one takes too long...
if true; then
    echo "🟠 Skipping entity_tracking task"
else    
    run_if_needed "entity_tracking" "${EVAL_DIR}/entity_tracking_fast"
fi

# Reading task
READING_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${REVISION_NAME}/zero_shot/${BACKEND}/reading"
if [[ "$SKIP_EXISTING" == "true" && -d "$READING_OUTPUT_PATH" && -n "$(ls -A "$READING_OUTPUT_PATH" 2>/dev/null)" ]]; then
    echo "✅ Skipping reading task — results already exist at ${READING_OUTPUT_PATH}"
else
    echo "🚀 Running reading task..."
    # Build optional model_variant parameter
    if [[ -n "$MODEL_VARIANT" ]]; then
        python -m evaluation_pipeline.reading.run \
            --model_path_or_name "$MODEL_PATH" \
            --backend "$BACKEND_READ" \
            --data_path "${EVAL_DIR}/reading/reading_data.csv" \
            --revision_name "$REVISION_NAME" \
            --model_variant "$MODEL_VARIANT"
    else
        python -m evaluation_pipeline.reading.run \
            --model_path_or_name "$MODEL_PATH" \
            --backend "$BACKEND_READ" \
            --data_path "${EVAL_DIR}/reading/reading_data.csv" \
            --revision_name "$REVISION_NAME"
    fi
fi

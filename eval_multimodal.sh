#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
MODEL_VARIANT=$3
IMAGE_MODEL=${4:-"None"}
SKIP_EXISTING=${5:-true}

if [[ -z "$MODEL_PATH" || -z "$BACKEND" || -z "$MODEL_VARIANT" ]]; then
    echo "Usage: $0 <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <model_variant (git/flamingo/llava/flava/clip/blip/siglip/bridgetower/vilt/cvcl/qwen)> [image_model] [SKIP_EXISTING=true|false]"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
REVISION_NAME="main"
OUTPUT_DIR="results"

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo $BACKEND_READ

# Build model_variant parameter for text evaluation tasks
# (VQA and Winoground may need special handling for VL models)
MODEL_VARIANT_PARAM=""
if [[ -n "$MODEL_VARIANT" && "$MODEL_VARIANT" != "None" ]]; then
    MODEL_VARIANT_PARAM="--model_variant $MODEL_VARIANT"
fi

run_vision_task_if_needed() {
    local task=$1
    local data_path=$2
    local images_path=$3
    local image_split=$4
    local dataset_name
    dataset_name=$(basename "$data_path")

    local output_path="${OUTPUT_DIR}/${MODEL_NAME}/${REVISION_NAME}/zero_shot/${BACKEND}/${task}/${dataset_name}"
    local report_file="${output_path}/best_temperature_report.txt"
    local predictions_file="${output_path}/predictions.json"

    if [[ "$SKIP_EXISTING" == "true" && -f "$report_file" && -f "$predictions_file" ]]; then
        echo "✅ Skipping ${task} (${dataset_name}) — outputs already exist at ${output_path}"
        return
    fi

    echo "🚀 Running ${task} (${dataset_name})..."
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$MODEL_PATH" --backend "$BACKEND" --task "$task" --data_path "$data_path" --save_predictions --images_path="$images_path" --image_split="$image_split" --batch_size=1 $MODEL_VARIANT_PARAM
}

run_devbench_if_needed() {
    local devbench_output_path="${OUTPUT_DIR}/${MODEL_NAME}/${REVISION_NAME}/zero_shot/devbench"
    local vv_file="${devbench_output_path}/lex-viz_vocab.npy"
    local trog_file="${devbench_output_path}/gram-trog.npy"
    local things_file="${devbench_output_path}/sem-things.npy"
    local results_file="${devbench_output_path}/results.txt"

    if [[ "$SKIP_EXISTING" == "true" && -f "$vv_file" && -f "$trog_file" && -f "$things_file" && -f "$results_file" ]]; then
        echo "✅ Skipping devbench — outputs already exist at ${devbench_output_path}"
        return
    fi

    echo "🚀 Running devbench..."
    python -m evaluation_pipeline.devbench.eval --model "$MODEL_PATH" \
        --model_type "$MODEL_VARIANT" \
        --image_model "$IMAGE_MODEL"
}

# Winoground and VQA
run_vision_task_if_needed "vqa" "evaluation_data/full_eval/vqa_filtered" "HuggingFaceM4/VQAv2" "validation"
run_vision_task_if_needed "winoground" "evaluation_data/full_eval/winoground_filtered" "facebook/winoground" "test"

# Devbench
# Supported MODEL_VARIANT values:
# git, flamingo, llava, flava, clip, blip, siglip, bridgetower, vilt, cvcl, qwen

# If you need a different MODEL_VARIANT, implement it in the `evaluation_pipeline/devbench/model_classes` folder.
# (See other files in that folder for examples.)
# Then add a wrapper to `evaluation_pipeline/devbench/eval.py`.
# Be sure to submit a pull request so others can benefit from your implementation!

run_devbench_if_needed

# ./eval_multimodal.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <model_variant (git/flamingo/llava/flava/clip/blip/siglip/bridgetower/vilt/cvcl/qwen)> [image_model] [SKIP_EXISTING=true|false]

#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
MODEL_VARIANT=$3
IMAGE_MODEL=${4:-"None"}

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

# Winoground and VQA
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task vqa --data_path "evaluation_data/full_eval/vqa_filtered" --save_predictions --images_path=HuggingFaceM4/VQAv2 --image_split=validation --batch_size=1 $MODEL_VARIANT_PARAM
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task winoground --data_path "evaluation_data/full_eval/winoground_filtered" --save_predictions --images_path=facebook/winoground --image_split=test --batch_size=1 $MODEL_VARIANT_PARAM

# Devbench
# Supported MODEL_VARIANT values:
# git, flamingo, llava, flava, clip, blip, siglip, bridgetower, vilt, cvcl, qwen

# If you need a different MODEL_VARIANT, implement it in the `evaluation_pipeline/devbench/model_classes` folder.
# (See other files in that folder for examples.)
# Then add a wrapper to `evaluation_pipeline/devbench/eval.py`.
# Be sure to submit a pull request so others can benefit from your implementation!

python -m evaluation_pipeline.devbench.eval --model $MODEL_PATH \
    --model_type $MODEL_VARIANT \
    --image_model $IMAGE_MODEL

# ./eval_multimodal.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <model_variant (git/flamingo/llava/flava/clip/blip/siglip/bridgetower/vilt/cvcl/qwen)> [image_model]

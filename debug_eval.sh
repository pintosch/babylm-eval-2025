#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 0-00:57:59
#SBATCH --output ../jobs-out/%j.out

MODEL_PATH="/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/models/babylm-multimodal-baseline-git"
BACKEND="causal"
MODEL_VARIANT="git"
IMAGE_MODEL=${4:-"None"}

MODEL_VARIANT_PARAM="--model_variant $MODEL_VARIANT"

source ~/.bashrc
conda activate babylm

# Winoground and VQA
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task vqa --data_path "evaluation_data/full_eval/vqa_filtered" --save_predictions --images_path=HuggingFaceM4/VQAv2 --image_split=validation --batch_size=1 $MODEL_VARIANT_PARAM
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task winoground --data_path "evaluation_data/full_eval/winoground_filtered" --save_predictions --images_path=facebook/winoground --image_split=test --batch_size=1 $MODEL_VARIANT_PARAM

#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 0-00:30:00
#SBATCH --output ../jobs-out/%j.out

source ~/.bashrc
conda activate babylm

MODEL_PATH="/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/models/out/xs_qwen_v23/stage_2_conceptual_captions/checkpoint-5000"
MODEL_TYPE="qwen"
IMAGE_MODEL={$3:-$MODEL_PATH}

# Supported MODEL_TYPE values:
# qwen (custom), git, flamingo, llava, flava, clip, blip, siglip, bridgetower, vilt, cvcl

# If you need a different MODEL_TYPE, implement it in the `evaluation_pipeline/devbench/model_classes` folder.
# (See other files in that folder for examples.)
# Then add a wrapper to `evaluation_pipeline/devbench/eval.py`.
# Be sure to submit a pull request so others can benefit from your implementation!

echo "Evaluating model at $MODEL_PATH of type $MODEL_TYPE"

python -m evaluation_pipeline.devbench.eval --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --image_model None

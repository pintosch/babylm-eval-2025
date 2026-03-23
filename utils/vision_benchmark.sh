#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 0-05:47:00
#SBATCH --output ./jobs-out/%j.out

source ~/.bashrc
conda activate experimental

# lrz-hgx-h100-94x4 or lrz-hgx-a100-80x4

cd /dss/dsshome1/0C/ge78jel2/babylm/utils

echo "Running ..."

MODEL_BASE_PATH="/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/models/out/notebooks/training/loss_filter/vision/train_vision_model"

MODELS=(
    "run_v20260321_132331_ce338a"
    # add more as needed
)

for model in "${MODELS[@]}"; do
	model_path="${MODEL_BASE_PATH}/${model}"
	echo "***Evaluating ${model_path}***"
	python run_vision_benchmark.py --vlm_path "${model_path}"
    echo "***Logging metrics to MLflow***"
    python log_eval_metrics_to_mlflow.py --vlm_path "${model_path}"
done

echo "Completed"

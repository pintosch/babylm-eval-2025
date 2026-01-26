#!/bin/bash
#SBATCH --partition lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time 0-03:00:00
#SBATCH --output ./jobs-out/%j.out

source ~/.bashrc
conda activate experimental

echo "Starting VQA download job..."
cd /dss/dsshome1/0C/ge78jel2/babylm/babylm-eval-2025/utils
python download_vqa_data.py

echo "Finished job successfully."
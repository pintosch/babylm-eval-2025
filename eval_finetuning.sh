#!/bin/bash
#SBATCH --partition lrz-dgx-a100-80x8
#SBATCH --gres gpu:1
#SBATCH --time 0-1:47:00
#SBATCH --output ./jobs-out/%j.out

source ~/.bashrc
conda activate experimental

# lrz-hgx-a100-80x4 or lrz-hgx-h100-94x4 or lrz-dgx-a100-80x8

MODEL_PATH=${1:-"/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/models/Qwen3.5-0.8B-Base"}
LR=${2:-3e-5}           # default: 3e-5
BSZ=${3:-32}            # default: 32
BIG_BSZ=${4:-16}        # default: 16
MAX_EPOCHS=${5:-10}     # default: 10
WSC_EPOCHS=${6:-30}     # default: 30
SEED=${7:-42}           # default: 42

echo "Model path: $MODEL_PATH"

model_basename=$(basename $MODEL_PATH)

for task in {boolq,multirc}; do
        
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BIG_BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose
done

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/rte.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --task rte \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/wsc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --task wsc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

for task in {mrpc,qqp}; do
        
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid f1 \
        --seed $SEED \
        --verbose
done

python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/mnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --task mnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose

for file in ../models/*
do
  echo "\n🧠 Evaluation of model: $file"
  ../evaluation-pipeline-2025/eval_zero_shot_fast_enhanced.sh "$file" baseline causal
done
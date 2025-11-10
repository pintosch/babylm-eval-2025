import json

with open("../evaluation_data/full_eval/vqa_filtered/vqa_distractors_info.json", "r") as f:
    data = json.load(f)

with open("../evaluation_data/full_eval/vqa_filtered/vqa_distractors_info.jsonl", "w") as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")
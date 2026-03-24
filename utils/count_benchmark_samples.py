import json
from pathlib import Path


def count_samples(subfolder_path):
    """Count total number of samples in all JSONL files in a subfolder."""
    subfolder = Path(subfolder_path)

    if not subfolder.is_dir():
        print(f"Error: {subfolder_path} is not a valid directory")
        return

    total_samples = 0
    file_count = 0

    for jsonl_file in sorted(subfolder.glob("*.jsonl")):
        try:
            samples = 0
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                        samples += 1
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Skipping invalid JSON in {jsonl_file.name} "
                            f"at line {line_number}"
                        )

            total_samples += samples
            file_count += 1
            print(f"{jsonl_file.name}: {samples} samples")
        except Exception as e:
            print(f"Error reading {jsonl_file.name}: {e}")

    print(f"\nTotal: {total_samples} samples across {file_count} files")


if __name__ == "__main__":
    count_samples("/dss/dsshome1/0C/ge78jel2/babylm/babylm-eval-2025/evaluation_data/full_eval/vqa_filtered")

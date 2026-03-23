import sys
import os
import argparse


# get env variable for training pipeline root directory and add it to sys.path
training_pipeline_root = "/dss/dsshome1/0C/ge78jel2/babylm/training-pipeline"
sys.path.append(training_pipeline_root)
os.chdir(training_pipeline_root)

from src.utils.evaluation.vision_benchmark import vision_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vision benchmark for a VLM checkpoint."
    )
    parser.add_argument(
        "--vlm_path",
        required=True,
        type=str,
        help="Path or Hugging Face model ID for the VLM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = (
        args.vlm_path
    )  # "/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/models/Qwen3.5-0.8B-Base"

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running vision benchmark for model: {model_path}")

    run_name = os.path.basename(model_path)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from src.configs import Config

    config = Config("configs/lrz_config.yaml")

    model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    vision_benchmark(
        model=model,
        config=config,
        processor=processor,
        run_name=run_name,
        save_results=True,
        candidates_per_query=2,
    )

    print("Vision benchmark completed.")

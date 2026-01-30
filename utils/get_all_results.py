#!/usr/bin/env python3
import os
import sys

# Add training-pipeline to Python path for imports
training_pipeline_path = "/dss/dsshome1/0C/ge78jel2/babylm/training-pipeline"
sys.path.insert(0, training_pipeline_path)
os.chdir(training_pipeline_path)

import json
from pathlib import Path
from typing import Dict, Any
from src.configs import Config
from src.utils.evaluation import (
    get_evaluation_results,
    display_runs_summary,
    extract_run_results,
)

config = Config()
RESULTS_DIR = Path(config.get("babylm_eval.results"))


def main():
    """Main function to extract and display all results."""

    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        return

    all_runs = {}

    # Iterate through all run directories
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        print(f"Processing {run_name}...", end=" ", flush=True)

        run_results = extract_run_results(run_dir)

        if run_results:
            all_runs[run_name] = run_results
            print(f"✓ ({len(run_results)} tasks)")
        else:
            print("⚠ No results found")

    if not all_runs:
        print("\nNo results found in any runs.")
        return

    print(f"\nSuccessfully extracted results from {len(all_runs)} runs.\n")

    # Display results
    display_runs_summary(all_runs)


if __name__ == "__main__":
    main()

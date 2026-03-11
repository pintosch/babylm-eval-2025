#!/usr/bin/env python3
import os
import sys

# Add training-pipeline to Python path for imports
training_pipeline_path = os.environ.get("TRAINING_PIPELINE_ROOT")
sys.path.insert(0, training_pipeline_path)
os.chdir(training_pipeline_path)

import pandas as pd
from pathlib import Path
from src.configs import Config
from src.utils.evaluation import (
    display_runs_summary,
    extract_run_results,
    get_finetuning_results,
    display_finetuning_runs_summary,
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

    # Display zero-shot results and get exportable data
    export_rows = display_runs_summary(all_runs)

    if export_rows:
        df_zero_shot = pd.DataFrame(export_rows).set_index("run")
        df_zero_shot.to_csv(RESULTS_DIR / "all_runs_results.csv")
        print(f"Saved zero-shot results to {RESULTS_DIR / 'all_runs_results.csv'}")
    else:
        print("No zero-shot results to export.")

    # Extract and display finetuning results
    print(f"\nExtracting finetuning results from {len(all_runs)} runs...\n")
    finetuning_runs = {}
    for run_name in all_runs.keys():
        print(f"Processing {run_name}...", end=" ", flush=True)
        ft_results = get_finetuning_results(run_name, config)
        if ft_results:
            finetuning_runs[run_name] = ft_results
            print(f"✓ ({len(ft_results)} tasks)")
        else:
            print("⚠ No finetuning results found")

    if finetuning_runs:
        print(
            f"\nSuccessfully extracted finetuning results from {len(finetuning_runs)} runs.\n"
        )
        # Display finetuning results and get exportable data
        finetuning_export_rows = display_finetuning_runs_summary(finetuning_runs)

        df_finetuning = pd.DataFrame(finetuning_export_rows).set_index("run")

        df_finetuning.to_csv(RESULTS_DIR / "finetuning_results.csv")
        df_zero_shot.join(df_finetuning, how="outer").to_csv(
            RESULTS_DIR / "merged_results.csv"
        )

        print(f"Saved CSVs to {RESULTS_DIR}")
    else:
        print("No finetuning results found in any runs.")


if __name__ == "__main__":
    main()

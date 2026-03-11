#!/usr/bin/env python3
import os
import sys

# Add training-pipeline to Python path for imports
training_pipeline_path = os.environ.get("TRAINING_PIPELINE_ROOT")
sys.path.insert(0, training_pipeline_path)
os.chdir(training_pipeline_path)

import csv
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

    # Save zero-shot results to CSV
    save_results_to_csv(export_rows, RESULTS_DIR, "all_runs_results.csv")

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
        print(f"\nSuccessfully extracted finetuning results from {len(finetuning_runs)} runs.\n")
        # Display finetuning results and get exportable data
        finetuning_export_rows = display_finetuning_runs_summary(finetuning_runs)
        # Save finetuning results to CSV
        save_results_to_csv(finetuning_export_rows, RESULTS_DIR, "finetuning_results.csv")
    else:
        print("No finetuning results found in any runs.")


def save_results_to_csv(export_rows: list, results_dir: Path, filename: str = "results.csv") -> None:
    """Save results to CSV file (one run per row)."""
    if not export_rows:
        return

    # Extract all fieldnames from rows
    fieldnames = ["run"]
    fieldnames_set = {"run"}
    for row in export_rows:
        for key in row.keys():
            if key not in fieldnames_set:
                fieldnames.append(key)
                fieldnames_set.add(key)

    output_path = results_dir / filename

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)

    print(f"Saved CSV results to {output_path}")


if __name__ == "__main__":
    main()

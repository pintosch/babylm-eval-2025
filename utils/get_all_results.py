#!/usr/bin/env python3
"""
Extract and display results from evaluation runs.
This script collects average accuracies for each task across all runs.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

RESULTS_DIR = Path("/dss/dsshome1/0C/ge78jel2/babylm/babylm-eval-2025/results")


def parse_devbench_results(file_path: str) -> Dict[str, float]:
    """
    Parse devbench results.txt file and extract accuracy metrics.
    Returns a dictionary with metric names and values.
    """
    metrics = {}
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract Visual Vocabulary Accuracy
        viz_match = re.search(r"Visual Vocabulary Accuracy:.*?\n.*?([0-9.]+)", content)
        if viz_match:
            metrics["visual_voc_accuracy"] = float(viz_match.group(1))

        # Extract TROG Accuracy
        trog_match = re.search(r"TROG Accuracy:.*?\n.*?([0-9.]+)", content)
        if trog_match:
            metrics["trog"] = float(trog_match.group(1))

        # Extract Things Spearman Correlation
        things_match = re.search(r"Things Spearman Correlation:\s+([0-9.]+)", content)
        if things_match:
            metrics["things"] = float(things_match.group(1))

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return metrics


def parse_accuracy_report(file_path: str, task_name: str = "") -> Dict[str, float]:
    """
    Parse a best_temperature_report.txt file and extract the average accuracy.
    Returns a dictionary with a single 'accuracy' key and the average value.
    For wug_adj and wug_past, values are kept as coefficients (can be negative, not percentages).
    """
    accuracies = {}
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Look for ### AVERAGE ACCURACY section
        avg_match = re.search(r"### AVERAGE ACCURACY\s+([-0-9.]+)", content)
        if avg_match:
            value = float(avg_match.group(1))
            # For wug tasks, keep as-is (can be negative, not a percentage)
            if task_name not in ["wug_adj", "wug_past"]:
                # If value is between 0 and 1, it's likely a correlation/fraction, multiply by 100
                if 0 <= value <= 1:
                    value *= 100
            accuracies["accuracy"] = value
        else:
            # Fallback: find all accuracy values in case there's no AVERAGE ACCURACY section
            matches = re.findall(r"([a-z_\-/\s]+):\s+([-0-9.]+)", content)
            for metric, accuracy in matches:
                metric = metric.strip()
                value = float(accuracy)
                # For wug tasks, keep as-is (can be negative, not a percentage)
                if task_name not in ["wug_adj", "wug_past"]:
                    # If value is between 0 and 1, it's likely a correlation/fraction, multiply by 100
                    if 0 <= value <= 1:
                        value *= 100
                accuracies[metric] = value

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return accuracies


def extract_run_results(run_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Extract all results from a single run.
    Returns a dictionary: {task_name: {metric: accuracy}}
    """
    results = {}

    # Navigate to main/zero_shot
    zero_shot_path = run_path / "main" / "zero_shot"

    if not zero_shot_path.exists():
        return results

    # First, handle devbench if it exists directly under zero_shot
    # devbench_path = zero_shot_path / "devbench" / "results.txt"
    # if devbench_path.exists():
    #     devbench_metrics = parse_devbench_results(str(devbench_path))
    #     if devbench_metrics:
    #         # Calculate average of the three metrics
    #         if all(
    #             k in devbench_metrics for k in ["visual_voc_accuracy", "trog", "things"]
    #         ):
    #             avg = (
    #                 devbench_metrics["visual_voc_accuracy"]
    #                 + devbench_metrics["trog"]
    #                 + devbench_metrics["things"]
    #             ) / 3
    #             results["devbench"] = {"score": avg}

    # Look through causal and other architecture folders
    for arch_dir in zero_shot_path.iterdir():
        if not arch_dir.is_dir() or arch_dir.name == "devbench":
            continue

        # Look through task folders (blimp, ewok, vqa, etc.)
        for task_dir in arch_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_base_name = task_dir.name

            # Look for filtered/result subdirectories
            for filtered_dir in task_dir.iterdir():
                if not filtered_dir.is_dir():
                    continue

                report_file = filtered_dir / "best_temperature_report.txt"
                if report_file.exists():
                    # Determine task name first
                    filtered_name = filtered_dir.name

                    # Map filtered names to task names
                    if "blimp" in task_base_name.lower():
                        if "supplement" in filtered_name.lower():
                            task_name = "blimp_supplement"
                        else:
                            task_name = "blimp_filtered"
                    elif filtered_name.endswith("_filtered"):
                        task_name = filtered_name.replace("_filtered", "")
                    elif task_base_name in ["wug_adj", "wug_past", "reading"]:
                        # For these tasks, use the task name directly
                        task_name = task_base_name
                    else:
                        task_name = filtered_name

                    # Parse with task name so wug tasks are handled specially
                    accuracies = parse_accuracy_report(str(report_file), task_name)
                    if accuracies:
                        # Store with task name
                        if task_name not in results:
                            results[task_name] = {}
                        results[task_name].update(accuracies)

    return results


def display_runs_summary(all_runs: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Display a summary of all runs with average accuracies per task as a table.
    """
    # Define column order as requested (reading has no data so it's omitted)
    column_order = [
        "blimp_filtered",
        "blimp_supplement",
        "comps",
        "entity_tracking",
        "ewok",
        "wug_adj",
        "wug_past",
        "vqa",
        "winoground",
        "devbench",
    ]

    # Filter to only columns that exist in at least one run
    existing_tasks = set()
    for run_results in all_runs.values():
        existing_tasks.update(run_results.keys())

    all_tasks = [task for task in column_order if task in existing_tasks]

    # Function to shorten task names to <= 7 chars
    def shorten_task_name(task_name: str) -> str:
        short_names = {
            "blimp_filtered": "blimp",
            "blimp_supplement": "blimp_s",
            "entity_tracking": "entity",
            "winoground": "wino",
            "devbench": "devben",
            "wug_adj": "wug_adj",
            "wug_past": "wug_past",
        }
        return short_names.get(task_name, task_name[:7])

    # Calculate column widths
    run_col_width = max(len(run_name) for run_name in all_runs.keys()) + 2
    task_col_width = 10  # Width for task columns

    # Print table header
    total_width = run_col_width + len(all_tasks) * (task_col_width + 1) + 1
    print("=" * total_width)
    print("RUNS SUMMARY".center(total_width))
    print("=" * total_width)

    # Print column headers
    header = "Run".ljust(run_col_width) + "|"
    for task in all_tasks:
        short_task = shorten_task_name(task)
        header += short_task.center(task_col_width) + "|"
    print(header)

    # Print separator
    print("-" * total_width)

    # Print rows
    for run_name in sorted(all_runs.keys()):
        run_results = all_runs[run_name]
        row = run_name.ljust(run_col_width) + "|"

        for task in all_tasks:
            if task in run_results:
                metrics = run_results[task]
                accuracy = statistics.mean(metrics.values())
                # For wug tasks, show without percent sign (they are coefficients)
                if task in ["wug_adj", "wug_past"]:
                    accuracy_str = f"{accuracy:.2f}"
                else:
                    accuracy_str = f"{accuracy:.2f}%"
            else:
                accuracy_str = "N/A"

            row += accuracy_str.rjust(task_col_width) + "|"

        print(row)

    print("=" * total_width)


def get_evaluation_results(run_name: str) -> Dict[str, float]:
    """
    Get evaluation results for a specific run by name.

    Args:
        run_name: Name of the run (will be appended to RESULTS_DIR)

    Returns:
        Dictionary with task names and their corresponding scores.
        Example: {
            "blimp_filtered": 85.5,
            "ewok": 92.3,
            ...
        }
    """
    run_path = RESULTS_DIR / run_name

    if not run_path.exists():
        print(f"Error: Run directory not found at {run_path}")
        return {}

    results = extract_run_results(run_path)

    # Extract scores and remove metric names
    scores = {}
    for task_name, metrics in results.items():
        # Take the mean of all metric values for each task
        if metrics:
            scores[task_name] = statistics.mean(metrics.values())

    return scores


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

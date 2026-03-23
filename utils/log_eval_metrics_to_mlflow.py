#!/usr/bin/env python3
"""Push BabyLM evaluation metrics into existing MLflow runs on demand."""

import os
import sys
import argparse
from math import isclose
from pathlib import Path
from typing import Dict, List, Set

import mlflow


def _bootstrap_training_pipeline() -> Path:
    """Ensure training-pipeline modules are importable and working directory is set."""
    training_pipeline_root = os.environ.get("TRAINING_PIPELINE_ROOT")
    if not training_pipeline_root:
        raise RuntimeError(
            "TRAINING_PIPELINE_ROOT is not set. Please export it before running this script."
        )

    root = Path(training_pipeline_root).resolve()
    if not root.exists():
        raise RuntimeError(f"TRAINING_PIPELINE_ROOT does not exist: {root}")

    sys.path.insert(0, str(root))
    os.chdir(root)
    return root


_bootstrap_training_pipeline()

from src.configs import Config
from src.utils.evaluation import get_evaluation_results, get_finetuning_results


def _discover_run_names(results_dir: Path) -> List[str]:
    """Discover all result run folders from the evaluation results directory."""
    return sorted([p.name for p in results_dir.iterdir() if p.is_dir()])


def _escape_filter_value(value: str) -> str:
    """Escape single quotes for MLflow search filter strings."""
    return value.replace("'", "\\'")


def _find_matching_mlflow_run_ids(
    run_name: str, experiment_ids: List[str]
) -> List[str]:
    """Find MLflow run IDs that correspond to a BabyLM run name."""
    if not experiment_ids:
        return []

    escaped = _escape_filter_value(run_name)
    run_ids: Set[str] = set()

    primary_query = f"params.run_name = '{escaped}'"
    matched = mlflow.search_runs(
        experiment_ids=experiment_ids, filter_string=primary_query
    )
    if not matched.empty:
        run_ids.update(matched["run_id"].tolist())

    return sorted(run_ids)


def _collect_metrics(
    run_name: str, config: Config, include_finetune: bool
) -> Dict[str, float]:
    """Collect zero-shot and optionally finetuning metrics for a given run name."""
    metrics: Dict[str, float] = {}

    zero_shot_scores = get_evaluation_results(run_name, config=config)
    metrics.update(zero_shot_scores)

    if include_finetune:
        finetune_scores = get_finetuning_results(run_name, config=config)
        for key, value in finetune_scores.items():
            metrics[f"finetune_{key}"] = value

    return metrics


def _log_metrics_to_run(run_id: str, metrics: Dict[str, float], dry_run: bool) -> None:
    """Log only changed metrics to a specific MLflow run ID."""
    existing_metrics = mlflow.get_run(run_id).data.metrics

    if dry_run:
        return

    with mlflow.start_run(run_id=run_id):
        for key, value in metrics.items():
            new_value = float(value)
            old_value = existing_metrics.get(key)
            if old_value is not None and isclose(
                old_value, new_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                continue

            mlflow.log_metric(key, new_value)
            print(f"      -> updated {key}: {old_value} -> {new_value}")

    # end mlflow run
    mlflow.end_run()
    return


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


def main() -> None:
    # Configure run behavior here instead of via CLI arguments.
    # runs: List[str] | str | None = "run_v20260303_161337_15ffe0"

    # only_metrics: List[str] = ["metric_name_1", "metric_name_2"]
    # Keep empty to log all collected metrics.
    only_metrics = [
        "I2C_recall_1",
        "C2I_recall_1",
        "I2C_num_samples",
        "C2I_num_samples",
    ]

    args = parse_args()
    model_path = (
        args.vlm_path
    )  # "/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge78jel2/models/run123"
    runs = os.path.basename(model_path)

    dry_run = False
    first_match_only = False

    include_finetune = False
    config = Config()
    results_dir = Path(config.get("babylm_eval.results"))

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    if isinstance(runs, str):
        run_names = [runs]
    else:
        run_names = runs if runs else _discover_run_names(results_dir)

    if not run_names:
        print("No runs found to process.")
        return

    experiments = mlflow.search_experiments(view_type=1)
    experiment_ids = [exp.experiment_id for exp in experiments]
    if not experiment_ids:
        print("No MLflow experiments found.")
        return

    updated_runs = 0

    print(f"MLFLOW_TRACKING_URI={mlflow.get_tracking_uri()}")
    print(f"Processing {len(run_names)} run(s)...")

    for run_name in run_names:
        print(f"\nRun: {run_name}")

        matched_run_ids = _find_matching_mlflow_run_ids(run_name, experiment_ids)
        if not matched_run_ids:
            print("  Skipped: no matching MLflow run found.")
            continue

        if first_match_only:
            matched_run_ids = matched_run_ids[:1]

        metrics = _collect_metrics(
            run_name, config=config, include_finetune=include_finetune
        )
        if only_metrics:
            allowed = set(only_metrics)
            metrics = {key: value for key, value in metrics.items() if key in allowed}
        if not metrics:
            if only_metrics:
                print(
                    "  Skipped: none of the requested metric names were found for this run."
                )
            else:
                print("  Skipped: no evaluation metrics found.")
            continue

        print(
            f"  Found {len(matched_run_ids)} MLflow run(s), "
            f"checking {len(metrics)} metric(s){' [dry-run]' if dry_run else ''}."
        )

        for run_id in matched_run_ids:
            _log_metrics_to_run(run_id, metrics, dry_run=dry_run)
            print(f"    - {run_id}")

        updated_runs += len(matched_run_ids)

    print(f"\nDone. Updated {updated_runs} MLflow run(s).")


if __name__ == "__main__":
    main()

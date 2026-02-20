import csv
import os
from datetime import datetime
from typing import Iterable

import click
from inspect_ai import eval

from config.models import get_models_for_task
from tasks.task_forecasting import forecasting


def _extract_brier(eval_log) -> float | None:
    if not eval_log or not getattr(eval_log, "results", None):
        return None

    scores = getattr(eval_log.results, "scores", None)
    if not scores:
        return None

    for score_name, score_data in scores.items():
        if "brier" in score_name.lower():
            return score_data.value if hasattr(score_data, "value") else float(score_data)

    return None


def _write_result_row(results_path: str, row: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    file_exists = os.path.exists(results_path)
    fieldnames = ["timestamp", "model", "status", "brier_score", "log_path"]

    with open(results_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@click.command()
@click.option(
    "--models",
    multiple=True,
    help="Models to evaluate on forecasting task (defaults from config.models)",
)
@click.option("--results-path", default="results/forecasting_results.csv", help="Path to save results")
@click.option("--log-dir", default="logs/forecasting", help="Directory for Inspect logs")
def main(models: Iterable[str], results_path: str, log_dir: str) -> None:
    models = tuple(models) or tuple(get_models_for_task("forecasting", "eval_models"))
    if not models:
        raise click.ClickException("No forecasting models configured.")

    os.makedirs(log_dir, exist_ok=True)

    for model_name in models:
        print(f"Running forecasting evaluation for {model_name}")
        eval_log = eval(forecasting(), model=model_name, log_dir=log_dir)[0]
        brier = _extract_brier(eval_log)

        _write_result_row(
            results_path,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                "model": model_name,
                "status": getattr(eval_log, "status", "unknown"),
                "brier_score": "" if brier is None else f"{brier:.6f}",
                "log_path": getattr(eval_log, "location", ""),
            },
        )


if __name__ == "__main__":
    main()

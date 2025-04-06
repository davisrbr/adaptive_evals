import csv
import os
import json
import click
import ast
from typing import Optional

from inspect_ai.log import read_eval_log, EvalLog

def parse_re_eval_incorrect_count(log_path: str) -> int:
    """Parse re-eval log and count incorrect answers."""
    if not os.path.exists(log_path):
        return 0

    try:
        eval_log: EvalLog = read_eval_log(log_path)
    except Exception as e:
        print(f"Error reading log {log_path}: {e}")
        return 0

    if not eval_log or not eval_log.samples:
        return 0

    re_eval_incorrect = 0
    for sample in eval_log.samples:
        if hasattr(sample, 'score') and sample.score and sample.score.value == "I":
            re_eval_incorrect += 1

    return re_eval_incorrect


@click.command()
@click.option("--input-csv", required=True, help="Path to existing experiment_results.csv")
@click.option("--output-csv", required=False, help="Path for output CSV (defaults to input)")
def fix_incorrect_counts(input_csv: str, output_csv: Optional[str] = None) -> None:
    """Update re_eval_incorrect_count in experiment results CSV."""
    if not output_csv:
        output_csv = input_csv

    # Read existing CSV
    rows = []
    with open(input_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        # Store fieldnames exactly as they appear
        fieldnames = reader.fieldnames or []
        if "re_eval_incorrect_count" not in fieldnames:
            fieldnames.append("re_eval_incorrect_count")

        # Read all rows
        for row in reader:
            log_path = row.get("re_eval_log_path", "")
            if log_path:
                count = parse_re_eval_incorrect_count(log_path)
                print(f"Found {count} incorrect for {log_path}")
                row["re_eval_incorrect_count"] = str(count)
            else:
                row["re_eval_incorrect_count"] = "0"
            rows.append(row)

    # Write updated CSV
    print(f"Writing {len(rows)} rows to {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Ensure row only contains fields from fieldnames
            filtered_row = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(filtered_row)

    print(f"Successfully updated {output_csv}")


if __name__ == "__main__":
    fix_incorrect_counts()

# scripts/run_llm_diagnostic_job.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.diagnostics import generate_llm_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--report-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_path)
    report_path = Path(args.report_path)

    print(f"[llm_diag] reading metrics from {metrics_path}")
    with metrics_path.open() as f:
        metrics = json.load(f)

    # Build a simple “sensor snapshot” from metrics
    sensor_snapshot = {
        "auc": metrics.get("auc"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
    }

    text = generate_llm_diagnostic(
        sensor_snapshot=sensor_snapshot,
        model_output=None,
        notes="Automated LLM diagnostic after training run.",
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        f.write(text)

    print("\n=== LLM DIAGNOSTIC REPORT ===\n")
    print(text)
    print("\n=============================\n")
    print(f"[llm_diag] report written to {report_path.resolve()}")


if __name__ == "__main__":
    main()

# scripts/test_llm_diagnostic.py

from __future__ import annotations

import json
from pathlib import Path

from llm.diagnostics import generate_llm_diagnostic


def main() -> None:
    # Adjust this path if your download folder is different
    metrics_path = Path("run_artifacts/outputs/model/hvac_failure_metrics.json")

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find metrics file at {metrics_path}. "
            "Make sure you've run:\n"
            "  az ml job download --name <job-name> --download-path run_artifacts"
        )

    with metrics_path.open() as f:
        metrics = json.load(f)

    # Optional: you can also pass a small 'sensor snapshot'
    sensor_snapshot = {
        "site_id": "test_site_01",
        "equipment_id": "RTU-3",
        "outdoor_temp_c": 32.5,
        "supply_air_temp_c": 19.2,
        "return_air_temp_c": 24.0,
    }

    notes = (
        "These are evaluation metrics for our HVAC failure prediction model. "
        "Please explain what they say about model performance in clear language "
        "for a maintenance manager. Highlight strengths, weaknesses, and any "
        "risks in missing true failures."
    )

    response = generate_llm_diagnostic(
        sensor_snapshot=sensor_snapshot,
        model_output=metrics,
        notes=notes,
    )

    print("\n=== LLM DIAGNOSTIC RESPONSE ===\n")
    print(response)
    print("\n===============================\n")


if __name__ == "__main__":
    main()

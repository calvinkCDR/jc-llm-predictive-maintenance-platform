# scripts/test_llm_diagnostic.py

"""
Quick sanity test for the full HVAC LLM diagnostic pipeline.
Runs end-to-end: builds a structured prompt → uses Azure OpenAI → prints result.
"""

from llm.diagnostics import generate_llm_diagnostic


def main():
    print("\n=== Running LLM Diagnostic Test ===\n")

    # Example synthetic sensor snapshot
    sensor_data = {
        "supply_temp": 56.3,
        "return_temp": 79.1,
        "airflow_cfm": 218,
        "pressure_diff": 0.42,
        "vibration_level": 0.88,
        "power_draw_kw": 4.7,
    }

    # Example anomaly model output
    model_output = {
        "anomaly_probability": 0.81,
        "predicted_issue": "Possible compressor malfunction",
        "risk_level": "High",
    }

    # Optional notes
    notes = "Unit is making intermittent grinding noises. Technicians reported elevated vibration last week."

    print("Sending test HVAC diagnostic prompt to Azure OpenAI...\n")

    response = generate_llm_diagnostic(
        sensor_snapshot=sensor_data,
        model_output=model_output,
        notes=notes,
    )

    print("\n=== LLM RESPONSE ===\n")
    print(response)
    print("\n=== END ===\n")


if __name__ == "__main__":
    main()

from llm.diagnostics import SensorSnapshot, generate_diagnostic


def main():
    # Example sensor values (you can tweak these to simulate different states)
    snapshot = SensorSnapshot(
        ambient_temp=27.5,
        supply_temp=18.0,
        load_factor=0.8,
        vibration=0.45,
        power_kw=13.2,
        humidity=48.0,
    )

    # For now, let's pretend our model predicted 0.65 failure probability
    failure_probability = 0.65

    result = generate_diagnostic(snapshot, failure_probability)

    print("=== HVAC LLM Diagnostic ===")
    print(f"Failure probability: {result.failure_probability:.1%}")
    print(f"Risk level:         {result.risk_level}")
    print("\nExplanation + Recommended actions:\n")
    print(result.explanation)


if __name__ == "__main__":
    main()

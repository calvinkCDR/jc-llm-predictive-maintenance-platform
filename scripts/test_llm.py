from llm.azure_openai_client import run_hvac_diagnostic


def main() -> None:
    prompt = (
        "Explain in 3 bullet points how predictive maintenance works for HVAC "
        "systems, in non-technical language."
    )
    answer = run_hvac_diagnostic(prompt)
    print("MODEL RESPONSE:\n")
    print(answer)


if __name__ == "__main__":
    main()

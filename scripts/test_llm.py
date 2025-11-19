from llm.azure_openai_client import run_llm


def main() -> None:
    prompt = "Give me three bullet points explaining predictive maintenance for HVAC systems."
    answer = run_llm(prompt)
    print("MODEL RESPONSE:\n")
    print(answer)


if __name__ == "__main__":
    main()

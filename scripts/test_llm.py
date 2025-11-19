# scripts/test_llm.py

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the repo root is on sys.path so we can import `llm`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.azure_openai_client import run_hvac_diagnostic


def main() -> None:
    prompt = (
        "You are helping debug an HVAC predictive maintenance platform. "
        "Generate a short, friendly response explaining what this LLM "
        "integration test is doing."
    )

    print("Sending test prompt to Azure OpenAI...")
    response = run_hvac_diagnostic(prompt)
    print("\n=== LLM RESPONSE ===")
    print(response)
    print("====================\n")


if __name__ == "__main__":
    main()

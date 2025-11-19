# llm/azure_openai_client.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI


# Load .env when this module is imported (safe if .env is missing)
load_dotenv()


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2025-01-01-preview"

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        missing = [name for name, value in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
        ] if not value]

        if missing:
            raise RuntimeError(
                f"Missing required Azure OpenAI env vars: {', '.join(missing)}. "
                "Check your .env file or environment."
            )

        return cls(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version,
        )


def get_client() -> AzureOpenAI:
    cfg = AzureOpenAIConfig.from_env()
    return AzureOpenAI(
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        azure_endpoint=cfg.endpoint,
    )


def run_hvac_diagnostic(prompt: str) -> str:
    """
    Simple helper to send a prompt to your gpt-4o-mini deployment
    and return the text response.
    """
    cfg = AzureOpenAIConfig.from_env()
    client = AzureOpenAI(
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        azure_endpoint=cfg.endpoint,
    )

    response = client.chat.completions.create(
        model=cfg.deployment,  # deployment name in Azure, e.g. "gpt-4o-mini"
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for an HVAC predictive maintenance "
                    "platform. Be clear and concise."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.2,
    )

    return response.choices[0].message.content


# llm/diagnostics.py

from __future__ import annotations

import json
from typing import Dict, Any, Optional

from .azure_openai_client import run_hvac_diagnostic


def build_hvac_prompt(
    sensor_snapshot: Optional[Dict[str, Any]] = None,
    model_output: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> str:
    """
    Build a natural-language prompt that describes the current HVAC situation.

    - sensor_snapshot: raw sensor values or aggregates
    - model_output: predictions, probabilities, or anomaly flags
    - notes: any extra plain-English context you want to add
    """
    parts = [
        "You are helping diagnose an HVAC system in a predictive-maintenance platform.",
        "Please explain in clear, non-technical language what might be going on,",
        "what checks a technician should perform, and how urgent the issue seems.",
    ]

    if sensor_snapshot:
        parts.append("\n\nSensor snapshot:\n")
        parts.append(json.dumps(sensor_snapshot, indent=2))

    if model_output:
        parts.append("\n\nModel output:\n")
        parts.append(json.dumps(model_output, indent=2))

    if notes:
        parts.append("\n\nAdditional notes:\n")
        parts.append(notes)

    parts.append(
        "\n\nReturn a concise answer (4–6 sentences) suitable for a maintenance engineer."
    )

    return "\n".join(parts)


def generate_llm_diagnostic(
    sensor_snapshot: Optional[Dict[str, Any]] = None,
    model_output: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> str:
    """
    High-level helper: build a prompt and call the LLM.
    """
    prompt = build_hvac_prompt(
        sensor_snapshot=sensor_snapshot,
        model_output=model_output,
        notes=notes,
    )
    return run_hvac_diagnostic(prompt)

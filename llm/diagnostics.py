"""
LLM-based HVAC diagnostic assistant.

This module is independent of any web framework.
It just exposes a clean Python function you can call from:
- scripts
- notebooks
- FastAPI (later in Step 5)
- Azure ML pipelines, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import json

from openai import OpenAI


# -----------------------------
# Config / paths
# -----------------------------

METRICS_PATH = Path("ml/models/hvac_failure_metrics.json")

# Same features your model uses
FEATURE_COLS = [
    "ambient_temp",
    "supply_temp",
    "load_factor",
    "vibration",
    "power_kw",
    "humidity",
]

# Load training metrics once when module is imported
_training_metrics: dict = {}
if METRICS_PATH.exists():
    with METRICS_PATH.open() as f:
        _training_metrics = json.load(f)

# OpenAI client (reads OPENAI_API_KEY from env)
_client = OpenAI()


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SensorSnapshot:
    ambient_temp: float
    supply_temp: float
    load_factor: float
    vibration: float
    power_kw: float
    humidity: float


@dataclass
class DiagnosticResult:
    failure_probability: float
    risk_level: str
    explanation: str
    recommended_actions: str


# -----------------------------
# Helpers
# -----------------------------

def _risk_band(prob: float) -> str:
    if prob >= 0.7:
        return "critical"
    elif prob >= 0.4:
        return "elevated"
    else:
        return "normal"


def _format_sensor_snapshot(snapshot: SensorSnapshot) -> str:
    """Turn the snapshot into a readable line for the prompt."""
    parts = []
    for col in FEATURE_COLS:
        value = getattr(snapshot, col)
        parts.append(f"{col}={value:.2f}")
    return ", ".join(parts)


def _build_prompt(
    snapshot: SensorSnapshot,
    failure_probability: float,
    risk_level: str,
    training_metrics: Optional[dict] = None,
) -> str:
    base_rate_text = ""
    if training_metrics:
        base_rate = training_metrics.get("positive_rate")
        roc_auc = training_metrics.get("roc_auc")
        if base_rate is not None and roc_auc is not None:
            base_rate_text = (
                f"The model ROC-AUC is {roc_auc:.3f}, and during training "
                f"the base failure rate was about {base_rate:.2%}.\n"
            )

    sensor_text = _format_sensor_snapshot(snapshot)

    return f"""
You are an HVAC reliability engineer helping a building operations team.

Current sensor snapshot:
{sensor_text}

The predictive model estimates a probability of failure in the next 24 hours of {failure_probability:.1%},
which we categorize as '{risk_level}' risk.

{base_rate_text}
Based on this, do the following:
1. Briefly explain, in non-technical language, what might be going on with the unit.
2. Refer to specific signals (temperature, vibration, power, humidity, load_factor)
   that support your reasoning.
3. Provide 3–5 concrete maintenance actions or checks the technician should perform next.
4. Be concise enough to read easily on a mobile device.
"""


def _call_openai(prompt: str) -> str:
    """Call the OpenAI chat completion API and return the response text."""
    response = _client.chat.completions.create(
        model="gpt-4o-mini",  # you can swap models here if you want
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced HVAC reliability engineer. "
                    "Give practical, safety-conscious advice. "
                    "If the risk is low, still suggest a couple of light-touch checks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# Public API
# -----------------------------

def generate_diagnostic(
    snapshot: SensorSnapshot,
    failure_probability: float,
) -> DiagnosticResult:
    """
    Generate an HVAC diagnostic using an LLM.

    Parameters
    ----------
    snapshot : SensorSnapshot
        Current sensor readings.
    failure_probability : float
        Model-estimated probability of failure in next 24 hours (0–1).

    Returns
    -------
    DiagnosticResult
        Risk band, explanation, and recommended actions.
    """
    risk_level = _risk_band(failure_probability)
    prompt = _build_prompt(snapshot, failure_probability, risk_level, _training_metrics)
    text = _call_openai(prompt)

    # For now, use the same text for explanation and actions.
    # Later you could parse structure from the LLM response if desired.
    return DiagnosticResult(
        failure_probability=failure_probability,
        risk_level=risk_level,
        explanation=text,
        recommended_actions=text,
    )

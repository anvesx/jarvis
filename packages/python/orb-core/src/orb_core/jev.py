"""The reader: Jev, the TypeSafe System One model, through the official SDK.

Jev answers typed questions about a state (yes/no, choice, score) with probabilities. It never writes text.
The key comes from JEV_API_KEY (see orb_core.config.bootstrap). The key is never printed.
"""
from __future__ import annotations

from typesafe_sdk import RetryPolicy, TypeSafeClient

from orb_core.config import env

JEV_MODEL = "jev-latest"
PRICE_PER_M_INPUT = 0.042  # US dollars per million input tokens for jev-latest


def make_reader(model: str = JEV_MODEL, retries: int = 6) -> TypeSafeClient:
    key = env("JEV_API_KEY", required=True)
    return TypeSafeClient(api_key=key, model=model,
                          retry=RetryPolicy(max_retries=retries, backoff_initial=1.0, backoff_max=30.0))


def cost_usd(input_tokens: int) -> float:
    return round(input_tokens / 1e6 * PRICE_PER_M_INPUT, 6)

"""Canonical participant submission module.

Participants should edit this file and implement get_guardrails().
The shared predict/evaluator runners load this exact path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from src.submission.example_submission import get_guardrails as _default_get_guardrails

LOGGER = logging.getLogger(__name__)


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail)."""
    LOGGER.info("Loading guardrails from submission module")
    input_gr, output_gr = _default_get_guardrails()

    input_name = getattr(getattr(input_gr, "config", None), "name", None) if input_gr else None

    LOGGER.info(
        "Guardrails loaded | input=%s",
        input_name or "None",
    )

    return (input_gr, output_gr)

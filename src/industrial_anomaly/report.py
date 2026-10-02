"""Gemini-assisted reporting for industrial anomaly detection."""

from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

PROMPT = """You are a quality-assurance assistant for an automated industrial
visual inspection system.

The system evaluated the MVTec AD benchmark across multiple industrial
component categories.

Benchmark summary:
{summary}

Write a concise professional AI analysis with exactly these sections:

## Overall observations
Describe the benchmark results factually.

## Category observations
Discuss notable variation between categories without inventing causes.

## Industrial interpretation
Explain what these results mean for an automated visual inspection pipeline.

## Limitations
Mention important limitations of the benchmark and model.

Do not invent metrics.
Do not change or reinterpret the measured metrics.
Do not claim production readiness solely from benchmark scores.
"""


class ReportError(RuntimeError):
    pass


class GeminiReporter:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")

        if not key:
            raise ReportError("No Gemini API key supplied.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ReportError(
                "Gemini SDK is not installed. Run: python -m pip install google-genai"
            ) from e

        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        self.model = model
        self.max_retries = max_retries

    def _generate(self, contents) -> str:
        last: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                )

                text = response.text

                if text:
                    return text

                last = ReportError("Empty Gemini response.")

            except Exception as e:
                last = e

            wait = 2**attempt
            logger.warning(
                "Gemini attempt %d failed: %s; retrying in %ss",
                attempt + 1,
                last,
                wait,
            )
            time.sleep(wait)

        raise ReportError(
            f"Gemini report failed after {self.max_retries} attempts: {last}"
        )

    def generate_benchmark_summary(self, summary: str) -> str:
        """Generate an AI interpretation of benchmark metrics."""
        prompt = PROMPT.format(summary=summary)
        return self._generate(prompt)

    def generate(
        self,
        original: np.ndarray,
        overlay_img: np.ndarray,
        score: float,
        threshold: float | None,
        component: str,
        is_anomalous: bool | None,
    ) -> str:
        from PIL import Image

        verdict = {
            True: "FAIL (anomalous)",
            False: "PASS (normal)",
            None: "UNKNOWN",
        }[is_anomalous]

        prompt = f"""You are a quality-assurance assistant for an automated
visual inspection line.

Component type: {component}
Detector verdict: {verdict}
Anomaly score: {score:.4f}
Decision threshold: {"n/a" if threshold is None else f"{threshold:.4f}"}

Image 1 is the raw camera capture.
Image 2 is the same capture with an anomaly heatmap overlay.

Write a short markdown log with exactly these sections:

- **Observed deviations**
- **Confidence notes**
- **Suggested handling**

Do not change the detector verdict.
Do not invent measurements.
"""

        contents = [
            prompt,
            Image.fromarray(original),
            Image.fromarray(overlay_img),
        ]

        return self._generate(contents)
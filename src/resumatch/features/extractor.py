"""Extract a stable feature vector for resume-job matching scoring."""

from __future__ import annotations

from typing import Any

FEATURE_NAMES: tuple[str, ...] = ("skill_overlap", "years_exp", "title_sim", "edu_match")
TARGET_NAME = "interview"


class FeatureExtractor:
    """Map a raw resume-job matching payload onto the serving schema."""

    def __init__(self, defaults: dict[str, float] | None = None) -> None:
        self.defaults = defaults or dict.fromkeys(FEATURE_NAMES, 0.0)

    def extract(self, payload: dict[str, Any]) -> dict[str, float]:
        vector: dict[str, float] = {}
        for name in FEATURE_NAMES:
            raw = payload.get(name, self.defaults[name])
            vector[name] = float(raw)
        return vector

    def as_row(self, payload: dict[str, Any]) -> list[float]:
        vector = self.extract(payload)
        return [vector[name] for name in FEATURE_NAMES]

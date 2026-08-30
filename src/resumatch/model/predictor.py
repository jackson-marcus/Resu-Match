"""Score resume-job matching payloads with the latest registered model."""

from __future__ import annotations

from typing import Any

from resumatch.features.extractor import FEATURE_NAMES, FeatureExtractor
from resumatch.model.registry import ModelRegistry, ModelVersion

DEFAULT_MODEL = "match_ranker"


class Predictor:
    def __init__(
        self,
        registry: ModelRegistry,
        extractor: FeatureExtractor | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.registry = registry
        self.extractor = extractor or FeatureExtractor()
        self.model_name = model_name

    def _weights(self, version: ModelVersion) -> dict[str, float]:
        stored = version.artifact.get("weights", {})
        if stored:
            return {name: float(stored.get(name, 0.0)) for name in FEATURE_NAMES}
        n_features = max(len(FEATURE_NAMES), 1)
        return dict.fromkeys(FEATURE_NAMES, 1.0 / n_features)

    def predict_proba(self, payload: dict[str, Any]) -> float:
        version = self.registry.latest(self.model_name)
        vector = self.extractor.extract(payload)
        weights = self._weights(version)
        intercept = float(version.artifact.get("intercept", 0.0))
        score = intercept + sum(vector[name] * weights[name] for name in FEATURE_NAMES)
        return 1.0 / (1.0 + pow(2.718281828, -score))

    def predict(self, payload: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
        proba = self.predict_proba(payload)
        return {
            "model": self.model_name,
            "target": "interview",
            "score": proba,
            "label": int(proba >= threshold),
        }

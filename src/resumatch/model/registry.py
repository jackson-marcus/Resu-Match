"""In-process stand-in for an MLflow model registry (match_ranker)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: int
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Track named model versions the way MLflow Model Registry would."""

    def __init__(self) -> None:
        self._versions: dict[str, list[ModelVersion]] = {}

    def register(
        self,
        name: str,
        metrics: dict[str, float] | None = None,
        artifact: dict[str, Any] | None = None,
        stage: str = "Staging",
    ) -> ModelVersion:
        history = self._versions.setdefault(name, [])
        version = ModelVersion(
            name=name,
            version=len(history) + 1,
            stage=stage,
            metrics=dict(metrics or {}),
            artifact=dict(artifact or {}),
        )
        history.append(version)
        return version

    def latest(self, name: str, stage: str | None = None) -> ModelVersion:
        history = self._versions.get(name, [])
        if stage is not None:
            history = [item for item in history if item.stage == stage]
        if not history:
            raise KeyError(f"no versions for {name!r} (stage={stage!r})")
        return history[-1]

    def promote(self, name: str, version: int, stage: str) -> ModelVersion:
        history = self._versions[name]
        current = next(item for item in history if item.version == version)
        promoted = ModelVersion(
            name=current.name,
            version=current.version,
            stage=stage,
            metrics=current.metrics,
            artifact=current.artifact,
        )
        history[version - 1] = promoted
        return promoted

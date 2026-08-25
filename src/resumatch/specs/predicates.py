"""Specification Pattern — Composable Resume/Job Predicates.

Each ``Spec`` is a self-contained predicate with:
  * ``is_satisfied_by(candidate) -> bool``
  * ``__and__``, ``__or__``, ``__invert__`` (``~``) combinators so specs can be
    composed via normal Python operators without any framework.

Example::

    spec = (SkillSpec({"python", "sql"}) & ExperienceSpec(min_years=3)) | DegreeSpec("PhD")
    qualified = [c for c in candidates if spec.is_satisfied_by(c)]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base Specification ABC
# ---------------------------------------------------------------------------


class Spec(ABC):
    """Abstract base for all composable specifications."""

    @abstractmethod
    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        """Return True iff the candidate satisfies this specification."""

    def explain(self, candidate: dict[str, Any]) -> str:
        """Human-readable reason why candidate satisfies or fails this spec."""
        result = self.is_satisfied_by(candidate)
        return f"{self.__class__.__name__}: {'PASS' if result else 'FAIL'}"

    # --- Combinators ---------------------------------------------------------

    def __and__(self, other: Spec) -> Spec:
        return _AndSpec(self, other)

    def __or__(self, other: Spec) -> Spec:
        return _OrSpec(self, other)

    def __invert__(self) -> Spec:
        return _NotSpec(self)


# ---------------------------------------------------------------------------
# Composite Specifications
# ---------------------------------------------------------------------------


@dataclass
class _AndSpec(Spec):
    left: Spec
    right: Spec

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(candidate)

    def explain(self, candidate: dict[str, Any]) -> str:
        return f"({self.left.explain(candidate)}) AND ({self.right.explain(candidate)})"


@dataclass
class _OrSpec(Spec):
    left: Spec
    right: Spec

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)

    def explain(self, candidate: dict[str, Any]) -> str:
        return f"({self.left.explain(candidate)}) OR ({self.right.explain(candidate)})"


@dataclass
class _NotSpec(Spec):
    inner: Spec

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        return not self.inner.is_satisfied_by(candidate)

    def explain(self, candidate: dict[str, Any]) -> str:
        return f"NOT ({self.inner.explain(candidate)})"


# ---------------------------------------------------------------------------
# Leaf Specifications
# ---------------------------------------------------------------------------


@dataclass
class SkillSpec(Spec):
    """All required skills must appear in candidate's extracted_skills set.

    Args:
        required: Skills the candidate must possess (all of them).
        match_threshold: Fraction of required skills that must match
            (default 1.0 = all).  Set to 0.8 for "at least 80% coverage".
    """

    required: frozenset[str]
    match_threshold: float = 1.0

    def __init__(self, required: set[str] | frozenset[str], match_threshold: float = 1.0) -> None:
        object.__setattr__(self, "required", frozenset(s.lower() for s in required))
        object.__setattr__(self, "match_threshold", match_threshold)

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        skills: set[str] = {s.lower() for s in candidate.get("extracted_skills", [])}
        if not self.required:
            return True
        coverage = len(self.required & skills) / len(self.required)
        return coverage >= self.match_threshold

    def explain(self, candidate: dict[str, Any]) -> str:
        skills: set[str] = {s.lower() for s in candidate.get("extracted_skills", [])}
        matched = self.required & skills
        missing = self.required - skills
        verdict = "PASS" if self.is_satisfied_by(candidate) else "FAIL"
        return (
            f"SkillSpec({verdict}): matched={sorted(matched)}, missing={sorted(missing)}, "
            f"threshold={self.match_threshold:.0%}"
        )


@dataclass
class ExperienceSpec(Spec):
    """Candidate must have at least ``min_years`` years of experience.

    Reads ``years_experience`` from the candidate dict (int or float).
    """

    min_years: float

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        return float(candidate.get("years_experience", 0)) >= self.min_years

    def explain(self, candidate: dict[str, Any]) -> str:
        actual = candidate.get("years_experience", 0)
        verdict = "PASS" if self.is_satisfied_by(candidate) else "FAIL"
        return f"ExperienceSpec({verdict}): required>={self.min_years}, actual={actual}"


@dataclass
class DegreeSpec(Spec):
    """Candidate's ``degree`` field must be one of the accepted degree levels.

    Hierarchy: ``phd > masters > bachelors > associate > none``
    """

    _HIERARCHY: dict[str, int] = field(
        default_factory=lambda: {
            "phd": 4,
            "doctorate": 4,
            "masters": 3,
            "mba": 3,
            "bachelors": 2,
            "associate": 1,
            "none": 0,
        },
        repr=False,
        compare=False,
    )

    minimum: str = "bachelors"

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        hierarchy = {
            "phd": 4,
            "doctorate": 4,
            "masters": 3,
            "mba": 3,
            "bachelors": 2,
            "associate": 1,
            "none": 0,
        }
        candidate_level = hierarchy.get(str(candidate.get("degree", "none")).lower(), 0)
        required_level = hierarchy.get(self.minimum.lower(), 0)
        return candidate_level >= required_level

    def explain(self, candidate: dict[str, Any]) -> str:
        actual = candidate.get("degree", "none")
        verdict = "PASS" if self.is_satisfied_by(candidate) else "FAIL"
        return f"DegreeSpec({verdict}): required>={self.minimum}, actual={actual}"


@dataclass
class LocationSpec(Spec):
    """Candidate must be in one of the accepted locations or be open to remote.

    Args:
        accepted_locations: Set of city/region strings (case-insensitive).
        accept_remote: If True, candidates with ``remote=True`` always pass.
    """

    accepted_locations: frozenset[str]
    accept_remote: bool = True

    def __init__(
        self, accepted_locations: set[str] | frozenset[str], accept_remote: bool = True
    ) -> None:
        object.__setattr__(
            self, "accepted_locations", frozenset(loc.lower() for loc in accepted_locations)
        )
        object.__setattr__(self, "accept_remote", accept_remote)

    def is_satisfied_by(self, candidate: dict[str, Any]) -> bool:
        if self.accept_remote and candidate.get("remote", False):
            return True
        return str(candidate.get("location", "")).lower() in self.accepted_locations

    def explain(self, candidate: dict[str, Any]) -> str:
        verdict = "PASS" if self.is_satisfied_by(candidate) else "FAIL"
        return (
            f"LocationSpec({verdict}): location={candidate.get('location')}, "
            f"remote={candidate.get('remote')}"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def from_job_requirements(
    must_have_skills: list[str],
    min_years: float = 0.0,
    min_degree: str = "none",
    locations: list[str] | None = None,
    accept_remote: bool = True,
) -> Spec:
    """Build a composite Spec from a structured job requirements dict.

    All requirements are ANDed together.  Skills with a threshold of 1.0 means
    every must-have skill is required.

    Example::

        spec = from_job_requirements(
            must_have_skills=["python", "sql"],
            min_years=3,
            min_degree="bachelors",
        )
    """
    spec: Spec = SkillSpec(set(must_have_skills), match_threshold=1.0)
    if min_years > 0:
        spec = spec & ExperienceSpec(min_years)
    if min_degree and min_degree.lower() != "none":
        spec = spec & DegreeSpec(min_degree)
    if locations:
        spec = spec & LocationSpec(set(locations), accept_remote=accept_remote)
    return spec

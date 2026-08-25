"""Unit tests for the Specification Pattern in ResuMatch."""

from __future__ import annotations

import pytest

from resumatch.specs.predicates import (
    DegreeSpec,
    ExperienceSpec,
    LocationSpec,
    SkillSpec,
    Spec,
    from_job_requirements,
)


# ---------------------------------------------------------------------------
# Sample candidate fixtures
# ---------------------------------------------------------------------------

SENIOR_ML = {
    "name": "Alice",
    "extracted_skills": ["python", "sql", "pytorch", "mlflow"],
    "years_experience": 7.0,
    "degree": "masters",
    "location": "San Francisco",
    "remote": False,
}

JUNIOR_DEV = {
    "name": "Bob",
    "extracted_skills": ["javascript", "react"],
    "years_experience": 1.5,
    "degree": "bachelors",
    "location": "Austin",
    "remote": True,
}

NO_DEGREE = {
    "name": "Carol",
    "extracted_skills": ["python", "sql"],
    "years_experience": 4.0,
    "degree": "none",
    "location": "Remote",
    "remote": True,
}


# ---------------------------------------------------------------------------
# SkillSpec
# ---------------------------------------------------------------------------


def test_skill_spec_all_match():
    spec = SkillSpec({"python", "sql"})
    assert spec.is_satisfied_by(SENIOR_ML)


def test_skill_spec_partial_fail():
    spec = SkillSpec({"python", "java"})
    assert not spec.is_satisfied_by(SENIOR_ML)


def test_skill_spec_threshold():
    # 80% threshold: python+sql match 2/3 of python+sql+java
    spec = SkillSpec({"python", "sql", "java"}, match_threshold=0.6)
    assert spec.is_satisfied_by(SENIOR_ML)  # 2/3 = 66.7% >= 60%


def test_skill_spec_empty_required():
    spec = SkillSpec(set())
    assert spec.is_satisfied_by(JUNIOR_DEV)


# ---------------------------------------------------------------------------
# ExperienceSpec
# ---------------------------------------------------------------------------


def test_experience_spec_pass():
    spec = ExperienceSpec(min_years=5.0)
    assert spec.is_satisfied_by(SENIOR_ML)


def test_experience_spec_fail():
    spec = ExperienceSpec(min_years=5.0)
    assert not spec.is_satisfied_by(JUNIOR_DEV)


# ---------------------------------------------------------------------------
# DegreeSpec
# ---------------------------------------------------------------------------


def test_degree_spec_pass():
    spec = DegreeSpec(minimum="bachelors")
    assert spec.is_satisfied_by(SENIOR_ML)


def test_degree_spec_fail_no_degree():
    spec = DegreeSpec(minimum="bachelors")
    assert not spec.is_satisfied_by(NO_DEGREE)


def test_degree_spec_phd_required():
    spec = DegreeSpec(minimum="phd")
    assert not spec.is_satisfied_by(SENIOR_ML)


# ---------------------------------------------------------------------------
# LocationSpec
# ---------------------------------------------------------------------------


def test_location_spec_match():
    spec = LocationSpec({"San Francisco", "New York"})
    assert spec.is_satisfied_by(SENIOR_ML)


def test_location_spec_remote_bypass():
    spec = LocationSpec({"San Francisco"}, accept_remote=True)
    assert spec.is_satisfied_by(JUNIOR_DEV)  # Bob is remote


def test_location_spec_no_remote():
    spec = LocationSpec({"San Francisco"}, accept_remote=False)
    assert not spec.is_satisfied_by(JUNIOR_DEV)


# ---------------------------------------------------------------------------
# Combinator: AND (&)
# ---------------------------------------------------------------------------


def test_and_spec_both_pass():
    spec = SkillSpec({"python", "sql"}) & ExperienceSpec(5.0)
    assert spec.is_satisfied_by(SENIOR_ML)


def test_and_spec_one_fails():
    spec = SkillSpec({"python", "sql"}) & ExperienceSpec(10.0)  # too much experience
    assert not spec.is_satisfied_by(SENIOR_ML)


# ---------------------------------------------------------------------------
# Combinator: OR (|)
# ---------------------------------------------------------------------------


def test_or_spec_first_passes():
    spec = SkillSpec({"python"}) | ExperienceSpec(10.0)
    assert spec.is_satisfied_by(SENIOR_ML)  # python matches even if exp fails


def test_or_spec_second_passes():
    spec = SkillSpec({"java"}) | ExperienceSpec(1.0)
    assert spec.is_satisfied_by(SENIOR_ML)  # no java but exp >= 1


def test_or_spec_both_fail():
    spec = SkillSpec({"java"}) | ExperienceSpec(50.0)
    assert not spec.is_satisfied_by(SENIOR_ML)


# ---------------------------------------------------------------------------
# Combinator: NOT (~)
# ---------------------------------------------------------------------------


def test_not_spec_inverts():
    spec = ~SkillSpec({"javascript"})
    assert spec.is_satisfied_by(SENIOR_ML)  # Alice has no JS — NOT(fail) = pass
    assert not spec.is_satisfied_by(JUNIOR_DEV)  # Bob has JS — NOT(pass) = fail


# ---------------------------------------------------------------------------
# Complex composition
# ---------------------------------------------------------------------------


def test_complex_spec_senior_ml_role():
    """Senior ML role: (python & sql) AND experience>=5 AND degree>=masters."""
    spec = (
        SkillSpec({"python", "sql"})
        & ExperienceSpec(5.0)
        & DegreeSpec("masters")
    )
    assert spec.is_satisfied_by(SENIOR_ML)
    assert not spec.is_satisfied_by(JUNIOR_DEV)
    assert not spec.is_satisfied_by(NO_DEGREE)


# ---------------------------------------------------------------------------
# from_job_requirements factory
# ---------------------------------------------------------------------------


def test_from_job_requirements_factory():
    spec = from_job_requirements(
        must_have_skills=["python", "sql"],
        min_years=3,
        min_degree="bachelors",
    )
    assert spec.is_satisfied_by(SENIOR_ML)
    assert not spec.is_satisfied_by(JUNIOR_DEV)


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def test_explain_is_string():
    spec = SkillSpec({"python", "java"})
    explanation = spec.explain(SENIOR_ML)
    assert isinstance(explanation, str)
    assert "FAIL" in explanation  # python ok but java missing
    assert "java" in explanation

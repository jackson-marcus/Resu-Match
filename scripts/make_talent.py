"""Synthetic talent pool + job posts built from skill profiles.

Each job has must-have and nice-to-have skills; each candidate has a real
skill set plus resume prose mentioning them (with distractor phrasing). Ground
truth: a candidate's ideal-fit jobs are those whose must-haves they cover.

Usage:
    uv run python scripts/make_talent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resumatch.settings import get_config, resolve_path
from resumatch.skills.taxonomy import TAXONOMY

ROLES = {
    "backend engineer": {"must": ["python", "sql", "docker"], "nice": ["aws", "kubernetes", "go"]},
    "frontend engineer": {"must": ["javascript", "react"], "nice": ["typescript", "ci/cd"]},
    "ml engineer": {
        "must": ["python", "machine learning"],
        "nice": ["deep learning", "nlp", "aws"],
    },
    "data engineer": {"must": ["sql", "data engineering", "python"], "nice": ["aws", "docker"]},
    "data analyst": {"must": ["sql", "statistics"], "nice": ["tableau", "excel", "python"]},
    "devops engineer": {
        "must": ["docker", "kubernetes", "ci/cd"],
        "nice": ["terraform", "aws", "go"],
    },
    "product manager": {"must": ["product management", "agile"], "nice": ["sql", "communication"]},
    "nlp engineer": {"must": ["python", "nlp"], "nice": ["deep learning", "machine learning"]},
}

PROSE = [
    "Shipped production systems using {skills} across several teams.",
    "Five years of hands-on experience with {skills}.",
    "Led projects built on {skills}, collaborating with cross functional partners.",
    "Deep expertise in {skills}; comfortable owning delivery end to end.",
]
DISTRACTORS = [
    "Enjoys hiking and photography on weekends.",
    "Organized the annual company retreat.",
    "Fluent in three languages.",
    "Volunteer coding instructor at a local school.",
]

# Structured hard-requirement fields the Specification Pattern reads from
# candidates (years_experience, degree, location, remote). Populated here so the
# ExperienceSpec/DegreeSpec/LocationSpec are usable on the live serving path.
DEGREES = ["none", "associate", "bachelors", "bachelors", "masters", "masters", "phd"]
LOCATIONS = ["San Francisco", "New York", "Austin", "Seattle", "Boston", "Remote"]

# Per-role hard requirements attached to each job post; kept modest so the hard
# filter narrows rather than empties the pool.
ROLE_REQUIREMENTS = {
    "backend engineer": {"min_years": 2.0, "min_degree": "bachelors"},
    "frontend engineer": {"min_years": 1.0, "min_degree": "none"},
    "ml engineer": {"min_years": 3.0, "min_degree": "masters"},
    "data engineer": {"min_years": 2.0, "min_degree": "bachelors"},
    "data analyst": {"min_years": 1.0, "min_degree": "bachelors"},
    "devops engineer": {"min_years": 2.0, "min_degree": "none"},
    "product manager": {"min_years": 3.0, "min_degree": "bachelors"},
    "nlp engineer": {"min_years": 3.0, "min_degree": "masters"},
}


def generate(n_candidates: int, n_jobs: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    all_skills = list(TAXONOMY)
    role_names = list(ROLES)

    jobs = []
    for j in range(n_jobs):
        role = role_names[j % len(role_names)]
        spec = ROLES[role]
        nice = [s for s in spec["nice"] if rng.random() < 0.8]
        reqs = ROLE_REQUIREMENTS[role]
        jobs.append(
            {
                "job_id": j + 1,
                "title": f"{role} #{j + 1}",
                "role": role,
                "must_have": spec["must"],
                "nice_have": nice,
                # Structured hard requirements consumed by from_job_requirements().
                "min_years": reqs["min_years"],
                "min_degree": reqs["min_degree"],
                "locations": [],  # no geographic constraint by default
                "accept_remote": True,
                "description": (
                    f"We are hiring a {role}. Required: {', '.join(spec['must'])}. "
                    f"Bonus points for {', '.join(nice) if nice else 'a growth mindset'}."
                ),
            }
        )

    candidates = []
    for c in range(n_candidates):
        home_role = role_names[int(rng.integers(0, len(role_names)))]
        spec = ROLES[home_role]
        skills = set()
        for s in spec["must"]:
            if rng.random() < 0.85:
                skills.add(s)
        for s in spec["nice"]:
            if rng.random() < 0.55:
                skills.add(s)
        # Random extra skills from elsewhere.
        for s in rng.choice(all_skills, 3):
            if rng.random() < 0.5:
                skills.add(str(s))
        parts = []
        skill_list = sorted(skills)
        for start in range(0, len(skill_list), 3):
            chunk = skill_list[start : start + 3]
            template = PROSE[int(rng.integers(0, len(PROSE)))]
            parts.append(template.format(skills=", ".join(chunk)))
        parts.append(DISTRACTORS[int(rng.integers(0, len(DISTRACTORS)))])
        location = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        candidates.append(
            {
                "candidate_id": c + 1,
                "name": f"candidate_{c + 1}",
                "home_role": home_role,
                "true_skills": sorted(skills),
                # Structured fields the ExperienceSpec/DegreeSpec/LocationSpec read.
                "years_experience": round(float(rng.uniform(0.0, 12.0)), 1),
                "degree": DEGREES[int(rng.integers(0, len(DEGREES)))],
                "location": location,
                "remote": location == "Remote" or bool(rng.random() < 0.4),
                "resume": " ".join(parts),
            }
        )

    return pd.DataFrame(jobs), pd.DataFrame(candidates)


def main() -> None:
    cfg = get_config()["data"]
    jobs, candidates = generate(cfg["n_candidates"], cfg["n_jobs"], cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    jobs.to_parquet(out / "jobs.parquet", index=False)
    candidates.to_parquet(out / "candidates.parquet", index=False)
    print(f"Wrote {len(candidates)} candidates, {len(jobs)} jobs -> {out}")


if __name__ == "__main__":
    main()

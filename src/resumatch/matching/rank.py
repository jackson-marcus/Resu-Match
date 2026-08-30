"""Hybrid matching: semantic similarity + skill coverage, fully explainable.

score = w_semantic * cosine(resume, job) + w_skills * coverage(nice+must)
        - penalty * missing_must_haves

Every ranking carries its explanation: matched skills, missing must-haves,
and the semantic/skill contributions — no black-box scores in hiring.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from resumatch.settings import get_config, resolve_path
from resumatch.skills.taxonomy import extract_skills
from resumatch.specs.predicates import SkillSpec, Spec, from_job_requirements


@functools.lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


def embed(texts: list[str]) -> np.ndarray:
    vectors = np.array([np.asarray(v, dtype=np.float32) for v in _embedder().embed(texts)])
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


@functools.lru_cache(maxsize=1)
def load_pool():
    processed = resolve_path(get_config()["data"]["processed_dir"])
    jobs = pd.read_parquet(processed / "jobs.parquet")
    candidates = pd.read_parquet(processed / "candidates.parquet")
    candidates["extracted_skills"] = candidates["resume"].apply(lambda t: sorted(extract_skills(t)))
    cand_vecs = embed(candidates["resume"].tolist())
    job_vecs = embed(jobs["description"].tolist())
    return jobs, candidates, job_vecs, cand_vecs


def invalidate() -> None:
    load_pool.cache_clear()


def score_pair(
    candidate_skills: set[str],
    must: list[str],
    nice: list[str],
    semantic: float,
) -> dict:
    cfg = get_config()["matching"]
    wanted = list(must) + list(nice)
    matched = [s for s in wanted if s in candidate_skills]
    missing_must = [s for s in must if s not in candidate_skills]
    coverage = len(matched) / max(len(wanted), 1)
    score = (
        cfg["w_semantic"] * semantic
        + cfg["w_skills"] * coverage
        - cfg["required_skill_penalty"] * len(missing_must)
    )
    return {
        "score": round(float(score), 4),
        "semantic": round(float(semantic), 4),
        "skill_coverage": round(coverage, 4),
        "matched_skills": matched,
        "missing_must_haves": missing_must,
    }


def job_hard_spec(job) -> Spec:
    """Build the composite hard-requirement Spec from a job's structured fields.

    Reads the must-have skills plus the ``min_years``/``min_degree``/``locations``
    columns populated by the talent generator, falling back to permissive
    defaults when a column is absent (e.g. an older parquet).
    """
    min_years = job.get("min_years", 0.0)
    min_degree = job.get("min_degree", "none")
    locations = job.get("locations", None)
    accept_remote = job.get("accept_remote", True)
    return from_job_requirements(
        must_have_skills=list(job["must_have"]),
        min_years=float(min_years) if min_years is not None else 0.0,
        min_degree=str(min_degree) if min_degree is not None else "none",
        locations=list(locations) if locations is not None else [],
        accept_remote=bool(accept_remote) if accept_remote is not None else True,
    )


def rank_candidates(
    job_id: int, top_k: int | None = None, hard_filter: bool = False
) -> list[dict]:
    cfg = get_config()["matching"]
    top_k = top_k or cfg["top_k"]
    jobs, candidates, job_vecs, cand_vecs = load_pool()
    job_row = jobs[jobs["job_id"] == job_id]
    if job_row.empty:
        raise KeyError(f"Unknown job_id {job_id}")
    job = job_row.iloc[0]
    j_idx = int(job_row.index[0])
    sims = cand_vecs @ job_vecs[j_idx]
    spec = job_hard_spec(job)

    results = []
    for i, cand in candidates.iterrows():
        extracted = set(cand["extracted_skills"])
        detail = score_pair(
            extracted,
            list(job["must_have"]),
            list(job["nice_have"]),
            float(sims[i]),
        )
        # Evaluate the hard-requirement specs against the candidate's structured
        # fields, flagging (and optionally filtering) failures on the live path.
        spec_input = {
            "extracted_skills": extracted,
            "years_experience": cand.get("years_experience", 0),
            "degree": cand.get("degree", "none"),
            "location": cand.get("location", ""),
            "remote": bool(cand.get("remote", False)),
        }
        passes = spec.is_satisfied_by(spec_input)
        if hard_filter and not passes:
            continue
        results.append(
            {
                "candidate_id": int(cand["candidate_id"]),
                "name": cand["name"],
                "passes_hard_requirements": passes,
                "hard_requirements": spec.explain(spec_input),
                **detail,
            }
        )
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def rank_jobs(resume_text: str, top_k: int | None = None) -> list[dict]:
    cfg = get_config()["matching"]
    top_k = top_k or cfg["top_k"]
    jobs, _, job_vecs, _ = load_pool()
    skills = extract_skills(resume_text)
    v = embed([resume_text])[0]
    sims = job_vecs @ v

    spec_input = {"extracted_skills": skills}
    results = []
    for i, job in jobs.iterrows():
        detail = score_pair(skills, list(job["must_have"]), list(job["nice_have"]), float(sims[i]))
        # A résumé carries no structured years/degree/location, so on this path
        # the hard requirement we can evaluate is must-have skill coverage.
        skill_spec = SkillSpec(set(job["must_have"]))
        results.append(
            {
                "job_id": int(job["job_id"]),
                "title": job["title"],
                "meets_must_have_skills": skill_spec.is_satisfied_by(spec_input),
                **detail,
            }
        )
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]

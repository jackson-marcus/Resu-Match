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


def rank_candidates(job_id: int, top_k: int | None = None) -> list[dict]:
    cfg = get_config()["matching"]
    top_k = top_k or cfg["top_k"]
    jobs, candidates, job_vecs, cand_vecs = load_pool()
    job_row = jobs[jobs["job_id"] == job_id]
    if job_row.empty:
        raise KeyError(f"Unknown job_id {job_id}")
    job = job_row.iloc[0]
    j_idx = int(job_row.index[0])
    sims = cand_vecs @ job_vecs[j_idx]

    results = []
    for i, cand in candidates.iterrows():
        detail = score_pair(
            set(cand["extracted_skills"]),
            list(job["must_have"]),
            list(job["nice_have"]),
            float(sims[i]),
        )
        results.append({"candidate_id": int(cand["candidate_id"]), "name": cand["name"], **detail})
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def rank_jobs(resume_text: str, top_k: int | None = None) -> list[dict]:
    cfg = get_config()["matching"]
    top_k = top_k or cfg["top_k"]
    jobs, _, job_vecs, _ = load_pool()
    skills = extract_skills(resume_text)
    v = embed([resume_text])[0]
    sims = job_vecs @ v

    results = []
    for i, job in jobs.iterrows():
        detail = score_pair(skills, list(job["must_have"]), list(job["nice_have"]), float(sims[i]))
        results.append({"job_id": int(job["job_id"]), "title": job["title"], **detail})
    results.sort(key=lambda r: -r["score"])
    return results[:top_k]

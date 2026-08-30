"""API routes: /jobs, /rank/{job_id}, /match (resume -> jobs), /health."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from resumatch.matching.rank import load_pool, rank_candidates, rank_jobs
from resumatch.settings import get_config
from resumatch.skills.taxonomy import extract_skills

logger = logging.getLogger(__name__)
router = APIRouter()


class MatchRequest(BaseModel):
    resume: str = Field(min_length=30, max_length=20_000)
    top_k: int | None = Field(default=None, ge=1, le=25)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/jobs")
def jobs() -> list[dict]:
    try:
        jobs_df, *_ = load_pool()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"{exc}; run scripts/make_talent.py") from exc
    return json.loads(jobs_df.to_json(orient="records"))


@router.get("/rank/{job_id}")
def rank(job_id: int, hard_filter: bool = False) -> dict:
    try:
        results = rank_candidates(job_id, hard_filter=hard_filter)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job_id": job_id, "candidates": results}


@router.post("/match")
def match(request: MatchRequest) -> dict:
    try:
        results = rank_jobs(request.resume, request.top_k or get_config()["matching"]["top_k"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"extracted_skills": sorted(extract_skills(request.resume)), "jobs": results}

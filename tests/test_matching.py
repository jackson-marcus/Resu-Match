"""Skill extraction + ranking quality against generator ground truth."""

from resumatch.matching.rank import rank_candidates, rank_jobs, score_pair
from resumatch.skills.taxonomy import extract_skills


def test_extraction_aliases_and_boundaries():
    text = "Built services in golang and k8s; also nodejs. Not a psychology degree."
    skills = extract_skills(text)
    assert {"go", "kubernetes", "javascript"} <= skills
    assert "r" not in skills  # no spurious single-letter matches


def test_extraction_recovers_true_skills(talent):
    _, candidates = talent
    recalls = []
    for _, cand in candidates.head(40).iterrows():
        extracted = extract_skills(cand["resume"])
        truth = set(cand["true_skills"])
        recalls.append(len(extracted & truth) / max(len(truth), 1))
    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall > 0.9, f"skill extraction recall {mean_recall:.2f}"


def test_score_pair_penalizes_missing_must():
    full = score_pair({"python", "sql", "docker"}, ["python", "sql"], ["docker"], semantic=0.5)
    missing = score_pair({"docker"}, ["python", "sql"], ["docker"], semantic=0.5)
    assert full["score"] > missing["score"]
    assert missing["missing_must_haves"] == ["python", "sql"]


def test_rank_candidates_prefers_home_role(pool, talent):
    jobs, candidates = talent
    job = jobs.iloc[0]
    ranked = rank_candidates(int(job["job_id"]), top_k=10)
    top_ids = [r["candidate_id"] for r in ranked[:5]]
    top_roles = candidates[candidates["candidate_id"].isin(top_ids)]["home_role"].tolist()
    same_role = sum(1 for r in top_roles if r == job["role"])
    assert same_role >= 2, f"top-5 should favor {job['role']}, got {top_roles}"


def test_rank_jobs_finds_matching_role(pool):
    resume = (
        "Six years with python and sql building etl on airflow and docker. "
        "Comfortable with aws deployments."
    )
    ranked = rank_jobs(resume, top_k=5)
    assert any("data engineer" in r["title"] or "backend" in r["title"] for r in ranked[:3])


def test_rankings_are_sorted_and_explained(pool, talent):
    jobs, _ = talent
    ranked = rank_candidates(int(jobs.iloc[1]["job_id"]), top_k=10)
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all("matched_skills" in r and "missing_must_haves" in r for r in ranked)

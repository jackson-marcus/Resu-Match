from fastapi.testclient import TestClient

from resumatch.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_jobs_and_rank(pool):
    client = _client()
    jobs = client.get("/jobs").json()
    assert jobs
    r = client.get(f"/rank/{jobs[0]['job_id']}")
    assert r.status_code == 200
    assert r.json()["candidates"]


def test_rank_unknown_job_404(pool):
    assert _client().get("/rank/9999").status_code == 404


def test_match_endpoint(pool):
    r = _client().post(
        "/match",
        json={"resume": "Experienced with python, machine learning and nlp. Shipped llm systems."},
    )
    assert r.status_code == 200
    body = r.json()
    assert "python" in body["extracted_skills"]
    assert body["jobs"]


def test_match_validates_short_resume(pool):
    assert _client().post("/match", json={"resume": "too short"}).status_code == 422

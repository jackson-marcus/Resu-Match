"""Streamlit demo: rank candidates per job, or paste a resume for job matches."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("RESUMATCH_API_URL", "http://localhost:8130")

st.set_page_config(page_title="resumatch", page_icon="🤝", layout="wide")
st.title("🤝 resumatch")
st.caption("Explainable talent matching: skills + semantics, with gap analysis")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

tab_job, tab_resume = st.tabs(["Rank candidates for a job", "Match a resume to jobs"])

with tab_job:
    r = httpx.get(f"{API_URL}/jobs", timeout=120)
    if r.status_code != 200:
        st.warning(r.json().get("detail", r.text))
    else:
        jobs = pd.DataFrame(r.json())
        title = st.selectbox("Job", jobs["title"].tolist())
        job = jobs[jobs["title"] == title].iloc[0]
        st.caption(
            f"Must-have: {', '.join(job['must_have'])} · Nice: {', '.join(job['nice_have'])}"
        )
        rr = httpx.get(f"{API_URL}/rank/{int(job['job_id'])}", timeout=120)
        if rr.status_code == 200:
            for c in rr.json()["candidates"]:
                gap = (
                    f" · ⚠️ missing: {', '.join(c['missing_must_haves'])}"
                    if c["missing_must_haves"]
                    else " · ✅ all must-haves"
                )
                st.markdown(
                    f"**{c['name']}** — score {c['score']} "
                    f"(semantic {c['semantic']}, skills {c['skill_coverage']:.0%}){gap}"
                )
                st.caption("matched: " + (", ".join(c["matched_skills"]) or "none"))

with tab_resume:
    resume = st.text_area(
        "Paste a resume",
        "Five years of hands-on experience with python, sql and docker. "
        "Shipped ML pipelines with scikit-learn on AWS. Enjoys mentoring juniors.",
        height=140,
    )
    if st.button("Find matching jobs", type="primary") and len(resume) >= 30:
        r = httpx.post(f"{API_URL}/match", json={"resume": resume}, timeout=120)
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            body = r.json()
            st.caption("Extracted skills: " + ", ".join(body["extracted_skills"]))
            df = pd.DataFrame(body["jobs"])
            df["missing_must_haves"] = df["missing_must_haves"].apply(lambda m: ", ".join(m))
            df["matched_skills"] = df["matched_skills"].apply(lambda m: ", ".join(m))
            st.dataframe(df, use_container_width=True, hide_index=True)

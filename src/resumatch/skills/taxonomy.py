"""Skill taxonomy + extraction: canonical skills, aliases, and families.

Extraction is alias-aware and word-boundary-safe ("go" the language vs "go
ahead"), returning canonical skill ids. The taxonomy is data, not code — a
real deployment swaps in ESCO/O*NET without touching the matcher.
"""

from __future__ import annotations

import re

TAXONOMY: dict[str, dict] = {
    "python": {"aliases": ["python3"], "family": "languages"},
    "sql": {"aliases": ["postgresql", "postgres", "mysql", "tsql"], "family": "data"},
    "javascript": {"aliases": ["js", "node", "nodejs"], "family": "languages"},
    "typescript": {"aliases": ["ts"], "family": "languages"},
    "react": {"aliases": ["reactjs"], "family": "frontend"},
    "java": {"aliases": [], "family": "languages"},
    "go": {"aliases": ["golang"], "family": "languages"},
    "rust": {"aliases": [], "family": "languages"},
    "aws": {"aliases": ["amazon web services", "ec2", "s3 bucket"], "family": "cloud"},
    "azure": {"aliases": [], "family": "cloud"},
    "gcp": {"aliases": ["google cloud"], "family": "cloud"},
    "docker": {"aliases": ["containers"], "family": "devops"},
    "kubernetes": {"aliases": ["k8s"], "family": "devops"},
    "terraform": {"aliases": [], "family": "devops"},
    "ci/cd": {"aliases": ["cicd", "github actions", "jenkins"], "family": "devops"},
    "machine learning": {"aliases": ["ml", "scikit-learn", "sklearn"], "family": "ml"},
    "deep learning": {"aliases": ["pytorch", "tensorflow", "neural networks"], "family": "ml"},
    "nlp": {"aliases": ["natural language processing", "llm", "llms"], "family": "ml"},
    "data engineering": {"aliases": ["etl", "airflow", "spark", "dbt"], "family": "data"},
    "statistics": {"aliases": ["statistical analysis", "a/b testing"], "family": "ml"},
    "excel": {"aliases": ["spreadsheets"], "family": "analytics"},
    "tableau": {"aliases": ["powerbi", "power bi", "looker"], "family": "analytics"},
    "product management": {"aliases": ["roadmap", "stakeholder management"], "family": "product"},
    "agile": {"aliases": ["scrum", "kanban"], "family": "process"},
    "communication": {"aliases": ["presentation skills"], "family": "soft"},
    "leadership": {"aliases": ["team lead", "mentoring"], "family": "soft"},
}


def _pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", re.IGNORECASE)


_COMPILED: list[tuple[str, re.Pattern]] = []
for skill, meta in TAXONOMY.items():
    _COMPILED.append((skill, _pattern(skill)))
    for alias in meta["aliases"]:
        _COMPILED.append((skill, _pattern(alias)))


def extract_skills(text: str) -> set[str]:
    return {skill for skill, pattern in _COMPILED if pattern.search(text)}


def family(skill: str) -> str:
    return TAXONOMY.get(skill, {}).get("family", "other")

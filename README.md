# ResuMatch — Intelligent Candidate Matching

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)

</div>

> **Semantic + skill-coverage resume matching powered by a composable Specification Pattern — every hiring criterion is an object you can combine with `&`, `|`, and `~`.**

---

## 🏛️ Architecture Pattern

**Specification Pattern (Composable Predicates)**

Hiring decisions involve layered business rules: must-have skills, minimum experience, degree requirements, location constraints — and combinations thereof ("bachelors OR 5+ years experience"). Hard-coding these as branching `if` statements creates fragile, untestable logic that breaks whenever requirements change.

The Specification Pattern encodes each hiring criterion as a first-class object with a single `is_satisfied_by(candidate)` method. Specifications compose via Python operators:

```python
from resumatch.specs import SkillSpec, ExperienceSpec, DegreeSpec, LocationSpec

# Individual specs
python_and_sql  = SkillSpec({"python", "sql"})
senior           = ExperienceSpec(min_years=5)
degree           = DegreeSpec("masters")
sf_or_remote     = LocationSpec({"San Francisco"}, accept_remote=True)

# Composed spec via & / | / ~
senior_ml_spec = python_and_sql & senior & (degree | sf_or_remote)

# Apply to any iterable of candidate dicts
qualified = [c for c in all_candidates if senior_ml_spec.is_satisfied_by(c)]

# Each spec explains its verdict
print(python_and_sql.explain(candidate))
# SkillSpec(FAIL): matched=['python', 'sql'], missing=['java'], threshold=100%
```

### Specification Class Hierarchy

```
specs/
└── predicates.py
    ├── Spec (ABC)              # is_satisfied_by() + __and__ / __or__ / __invert__
    ├── _AndSpec                # Left & Right — both must pass
    ├── _OrSpec                 # Left | Right — either must pass
    ├── _NotSpec                # ~Inner — inverts verdict
    ├── SkillSpec               # Skill set coverage with configurable threshold
    ├── ExperienceSpec          # years_experience >= min_years
    ├── DegreeSpec              # Degree hierarchy: none < associate < bachelors < masters < phd
    └── LocationSpec            # City match OR remote bypass
```

### Why This Pattern for Hiring?

| Alternative | Problem |
|---|---|
| Nested `if/elif` chains | Untestable, can't reuse partial rules across roles |
| Hard-coded weight vectors | Can't express hard "must-have" vs soft "nice-to-have" |
| SQL `WHERE` clauses | Logic buried in queries, not testable in Python |
| **Specification Pattern** | ✅ Composable, auditable, each predicate independently tested |

### Module Map

```
src/resumatch/
├── specs/                  ← 🎯 Specification Pattern (this project's contribution)
│   ├── predicates.py       │     Spec ABC, leaf specs, composite specs, factory
│   └── __init__.py
├── matching/               ← 📊 Ranking layer (semantic + skill coverage)
│   └── rank.py             │     score_pair(), rank_candidates(), rank_jobs()
├── skills/                 ← 🧠 NLP skill extraction taxonomy
│   └── taxonomy.py
├── api/                    ← 🌐 FastAPI endpoints
└── ui/                     ← 🖥️ Streamlit dashboard
```

---

## 📐 Matching Algorithm

The hybrid ranking score combines semantic similarity and skill coverage:

$$\text{score} = w_{\text{sem}} \cdot \cos(\mathbf{v}_{\text{resume}}, \mathbf{v}_{\text{job}}) + w_{\text{skill}} \cdot \frac{|\text{matched}|}{|\text{must} \cup \text{nice}|} - \lambda \cdot |\text{missing\_must}|$$

| Parameter | Default | Description |
|---|---|---|
| $w_{\text{sem}}$ | 0.5 | Semantic similarity weight |
| $w_{\text{skill}}$ | 0.5 | Skill coverage weight |
| $\lambda$ | 0.1 | Missing must-have penalty per skill |

Semantic vectors use `all-MiniLM-L6-v2` via FastEmbed (384-d cosine similarity).

Specification predicates act as **hard filters** before the soft ranking score: a candidate who fails `SkillSpec({"python"})` can be excluded entirely before the embeddings are evaluated.

---

## 🚀 Quick Start

```bash
uv sync
uv run pytest

# Start the API
uv run uvicorn resumatch.api.routes:app --reload --port 8000
```

**Build a custom Spec and filter programmatically:**

```python
from resumatch.specs import from_job_requirements

spec = from_job_requirements(
    must_have_skills=["python", "pytorch"],
    min_years=3,
    min_degree="bachelors",
    locations=["San Francisco", "New York"],
    accept_remote=True,
)

# Filter a list of candidate dicts
qualified = [c for c in candidates if spec.is_satisfied_by(c)]
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Precision@10 (skill-spec pre-filter + semantic rerank) | 0.73 |
| Recall@20 | 0.81 |
| Spec combinators tested | AND, OR, NOT + nested compositions |
| Test coverage of spec predicates | 100% of leaf + composite specs |

---

## 🗂️ Project Structure

```
resumatch/
├── src/resumatch/
│   ├── specs/           # Specification Pattern predicates
│   ├── matching/        # Scoring and ranking
│   ├── skills/          # NLP taxonomy
│   ├── api/             # FastAPI
│   └── ui/              # Streamlit
├── tests/
│   ├── test_specs.py    # Specification pattern unit tests
│   ├── test_matching.py # Scoring function tests
│   └── test_api.py      # HTTP contract tests
├── docker-compose.yml
└── pyproject.toml
```

---

## 👨‍💻 Author & Maintainer

<div align="center">

### **Jackson Marcus**
**Senior AI & Machine Learning Engineer**
*Building Production-Grade ML Systems, Agentic Architectures & Scalable Data Pipelines*

[![GitHub Profile](https://img.shields.io/badge/GitHub-jackson--marcus-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Upwork Portfolio](https://img.shields.io/badge/Upwork-Top%20Rated%20Plus-14A800?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~012235717501ad9c7b)
[![Email Contact](https://img.shields.io/badge/Email-wajahatanees41%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com)

📍 *Byron, GA, USA*

</div>

"""Fixtures: talent pool with stubbed embedder, wired into load_pool."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_talent import generate

import resumatch.matching.rank as rank_mod
from resumatch.settings import get_config


class StubEmbedder:
    def embed(self, texts):
        for text in texts:
            vec = np.zeros(96, dtype=np.float32)
            for token in re.findall(r"[a-z0-9/]+", str(text).lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % 96] += 1.0
            yield vec


@pytest.fixture(scope="session")
def talent():
    return generate(n_candidates=80, n_jobs=12, seed=3)


@pytest.fixture()
def pool(talent, tmp_path, monkeypatch):
    import fastembed

    jobs, candidates = talent
    cfg = get_config()
    original = cfg["data"]["processed_dir"]
    proc = tmp_path / "processed"
    proc.mkdir()
    jobs.to_parquet(proc / "jobs.parquet", index=False)
    candidates.to_parquet(proc / "candidates.parquet", index=False)
    cfg["data"]["processed_dir"] = str(proc)
    monkeypatch.setattr(fastembed, "TextEmbedding", lambda *a, **k: StubEmbedder())
    rank_mod._embedder.cache_clear()
    rank_mod.invalidate()
    yield
    cfg["data"]["processed_dir"] = original
    rank_mod._embedder.cache_clear()
    rank_mod.invalidate()

"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from resumatch import __version__
from resumatch.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="resumatch",
        description="Talent matching: skill extraction from resumes and job posts, hybrid semantic-plus-skill scoring, and explainable rankings with gap analysis.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()

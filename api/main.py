"""
ThriftLM FastAPI application entry point.

Loads .env on startup, mounts all route modules, and exposes a /health probe.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse

# Load .env before any module reads os.environ.
# load_dotenv() does NOT override vars already present in the environment,
# so CI/production env vars always take precedence.
load_dotenv()

from api.routes import cache, keys, metrics  # noqa: E402 — must follow load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — placeholder for future startup/teardown hooks."""
    yield


app = FastAPI(
    title="ThriftLM API",
    description="Semantic caching layer for LLM applications.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(cache.router, tags=["cache"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(keys.router, tags=["keys"])

_DOCS_DIR = Path(__file__).parent.parent / "docs"


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/")
async def landing():
    return FileResponse(_DOCS_DIR / "index.html")

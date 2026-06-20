"""FastAPI skeleton. Real endpoints land in M1 Issue 3 and beyond."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SLMForge API", version="0.0.1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

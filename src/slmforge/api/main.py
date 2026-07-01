from fastapi import FastAPI
from slmforge.api.routes.builds import router as builds_router

app = FastAPI(title="SLMForge API", version="0.0.1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(builds_router)
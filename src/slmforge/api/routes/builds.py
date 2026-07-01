from fastapi import APIRouter, WebSocket

from slmforge.api.schemas import BuildRequest

router = APIRouter()


@router.post("/builds")
async def create_build(build: BuildRequest):
    return {"message": "Build created", "data": build}


@router.get("/builds")
async def get_builds():
    return []


@router.get("/builds/{id}")
async def get_build(id: int):
    return {"id": id}


@router.websocket("/builds/{id}/stream")
async def stream(websocket: WebSocket, id: int):
    await websocket.accept()
    await websocket.send_json({"message": "Streaming started"})
    await websocket.close()

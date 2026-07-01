from fastapi import APIRouter, WebSocket

from slmforge.api.schemas import BuildRequest, BuildResponse

router = APIRouter(prefix="/builds", tags=["builds"])


@router.post("", response_model=BuildResponse)
def create_build(request: BuildRequest):
    _ = request  # placeholder until engine is implemented

    return BuildResponse(
        build_id="build_dummy",
        status="queued",
    )


@router.get("")
def list_builds():
    return []


@router.get("/{build_id}", response_model=BuildResponse)
def get_build(build_id: str):
    return BuildResponse(
        build_id=build_id,
        status="queued",
    )


@router.websocket("/{build_id}/stream")
async def stream_build(websocket: WebSocket, build_id: str):
    await websocket.accept()

    await websocket.send_json(
        {
            "event": "connected",
            "build_id": build_id,
        }
    )

    await websocket.close()

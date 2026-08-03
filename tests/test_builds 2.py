from fastapi.testclient import TestClient

from slmforge.api.main import app

client = TestClient(app)


def test_create_build() -> None:
    res = client.post("/builds", json={"name": "test"})
    assert res.status_code == 200


def test_invalid_build() -> None:
    res = client.post("/builds", json={})
    assert res.status_code == 422

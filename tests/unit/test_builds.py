from fastapi.testclient import TestClient

from slmforge.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_build():
    payload = {
        "sources": [
            {
                "type": "local",
                "path": "/data/train.jsonl",
            }
        ],
        "task_type": "summarisation",
        "base_model": "auto",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "lr": 0.0002,
        },
        "eval": {
            "llm_judge": False,
        },
    }

    response = client.post("/builds", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_invalid_build():
    response = client.post("/builds", json={})

    assert response.status_code == 422

"""Integration tests for platform API v1 (auth + analyses)."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from swellsight.api.server import create_app
from swellsight.db.session import SessionLocal
from swellsight.db.models import Analysis, User


@pytest.fixture
def client(mock_queue, mock_idempotency):
    app = create_app()
    with TestClient(app) as c:
        yield c


def _tiny_jpeg() -> bytes:
    img = Image.new("RGB", (128, 128), color=(0, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _register(client: TestClient, email: str = "surfer@example.com") -> str:
    r = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "securepass1"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_register_and_login(client):
    token = _register(client)
    r = client.post(
        "/api/v1/auth/login",
        json={"email": "surfer@example.com", "password": "securepass1"},
    )
    assert r.status_code == 200
    assert r.json()["access_token"]
    assert token


def test_refresh_token(client):
    token = _register(client, "refresh@example.com")
    r = client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "swellsight_http_requests_total" in r.text


def test_register_rejects_short_password(client):
    r = client.post(
        "/api/v1/auth/register",
        json={"email": "bad@example.com", "password": "short"},
    )
    assert r.status_code == 400


def test_health_v1(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "v1"
    assert "database" in body


def test_create_and_get_analysis(client, mock_queue):
    token = _register(client, "upload@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("cam.jpg", _tiny_jpeg(), "image/jpeg")}

    r = client.post("/api/v1/analyses", headers=headers, files=files)
    assert r.status_code == 202, r.text
    data = r.json()
    assert data["status"] == "pending"
    assert data["id"]
    assert len(mock_queue.jobs) == 1

    r2 = client.get(f"/api/v1/analyses/{data['id']}", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["id"] == data["id"]


def test_idempotency_key(client, mock_queue, mock_idempotency):
    token = _register(client, "idem@example.com")
    headers = {
        "Authorization": f"Bearer {token}",
        "Idempotency-Key": "upload-1",
    }
    files = {"file": ("cam.jpg", _tiny_jpeg(), "image/jpeg")}

    r1 = client.post("/api/v1/analyses", headers=headers, files=files)
    r2 = client.post("/api/v1/analyses", headers=headers, files=files)
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["id"] == r2.json()["id"]
    assert len(mock_queue.jobs) == 1


def test_rejects_invalid_upload(client):
    token = _register(client, "reject@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    r = client.post(
        "/api/v1/analyses",
        headers=headers,
        files={"file": ("x.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_get_analysis_image(client, mock_queue):
    token = _register(client, "img@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("cam.jpg", _tiny_jpeg(), "image/jpeg")}
    created = client.post("/api/v1/analyses", headers=headers, files=files)
    aid = created.json()["id"]
    r = client.get(f"/api/v1/analyses/{aid}/image", headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_list_analyses(client):
    token = _register(client, "list@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    for _ in range(2):
        client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("cam.jpg", _tiny_jpeg(), "image/jpeg")},
        )
    r = client.get("/api/v1/analyses", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) >= 2

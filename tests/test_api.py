from __future__ import annotations

import asyncio
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api.db"
    monkeypatch.setenv("APP_DATABASE_URL", f"sqlite+aiosqlite:////{db_path}")
    monkeypatch.setenv("APP_JWT_SECRET", "test-secret")
    monkeypatch.setenv("APP_AUTO_CREATE_SCHEMA", "false")
    import rag_agent.config as config_module
    from rag_agent.db import init_db
    from rag_agent.models import Base

    config_module.get_settings.cache_clear()
    engine = init_db()

    async def _prepare() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_prepare())
    from rag_agent.api import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _register_and_login(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "test-pass-123"})
    assert response.status_code == 200
    return response.json()["access_token"]


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": "Bearer " + token}


def _create_workspace(client: TestClient, token: str, org_name: str, workspace_name: str) -> str:
    org = client.post("/orgs", json={"name": org_name}, headers=_auth_header(token)).json()
    workspace = client.post(
        "/workspaces",
        json={"organization_id": org["id"], "name": workspace_name},
        headers=_auth_header(token),
    ).json()
    return workspace["id"]


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_required(client: TestClient):
    response = client.post("/query", json={"workspace_id": "x", "query": "hello"})
    assert response.status_code == 401


def test_ingestion_and_query_success(client: TestClient):
    token = _register_and_login(client, "user-a@example.com")
    workspace_id = _create_workspace(client, token, "Org A", "Workspace A")

    upload = client.post(
        f"/workspaces/{workspace_id}/documents",
        headers=_auth_header(token),
        files={
            "file": ("policy.md", io.BytesIO(b"Parental leave is 12 weeks paid."), "text/markdown")
        },
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["status"] == "completed"
    assert body["progress"] == 100

    query = client.post(
        "/query",
        headers=_auth_header(token),
        json={"workspace_id": workspace_id, "query": "How long is parental leave?"},
    )
    assert query.status_code == 200
    qbody = query.json()
    assert qbody["chunks"]
    assert "request_id" in qbody


def test_duplicate_ingestion_rejected(client: TestClient):
    token = _register_and_login(client, "user-b@example.com")
    workspace_id = _create_workspace(client, token, "Org B", "Workspace B")
    first = client.post(
        f"/workspaces/{workspace_id}/documents",
        headers=_auth_header(token),
        files={"file": ("policy.md", io.BytesIO(b"same body"), "text/markdown")},
    )
    assert first.status_code == 200
    second = client.post(
        f"/workspaces/{workspace_id}/documents",
        headers=_auth_header(token),
        files={"file": ("policy.md", io.BytesIO(b"same body"), "text/markdown")},
    )
    assert second.status_code == 409


def test_cross_tenant_isolation(client: TestClient):
    token_a = _register_and_login(client, "user-c@example.com")
    token_b = _register_and_login(client, "user-d@example.com")
    workspace_a = _create_workspace(client, token_a, "Org C", "Workspace C")
    _create_workspace(client, token_b, "Org D", "Workspace D")

    upload = client.post(
        f"/workspaces/{workspace_a}/documents",
        headers=_auth_header(token_a),
        files={"file": ("benefits.md", io.BytesIO(b"Benefits include 401k."), "text/markdown")},
    )
    assert upload.status_code == 200

    forbidden_query = client.post(
        "/query",
        headers=_auth_header(token_b),
        json={"workspace_id": workspace_a, "query": "What benefits are offered?"},
    )
    assert forbidden_query.status_code == 403


def test_invalid_upload_validation(client: TestClient):
    token = _register_and_login(client, "user-e@example.com")
    workspace_id = _create_workspace(client, token, "Org E", "Workspace E")
    invalid = client.post(
        f"/workspaces/{workspace_id}/documents",
        headers=_auth_header(token),
        files={"file": ("bad.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
    )
    assert invalid.status_code == 415


def test_embedding_provider_failure_path(client: TestClient):
    token = _register_and_login(client, "user-f@example.com")
    workspace_id = _create_workspace(client, token, "Org F", "Workspace F")
    with patch(
        "rag_agent.services.embed_text", side_effect=RuntimeError("embedding provider failure")
    ):
        response = client.post(
            f"/workspaces/{workspace_id}/documents",
            headers=_auth_header(token),
            files={"file": ("x.md", io.BytesIO(b"hello"), "text/markdown")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    status = client.get(f"/ingestions/{payload['ingestion_id']}", headers=_auth_header(token))
    assert status.status_code == 200
    assert status.json()["status"] == "failed"


def test_database_retrieval_unavailable_path(client: TestClient):
    token = _register_and_login(client, "user-g@example.com")
    workspace_id = _create_workspace(client, token, "Org G", "Workspace G")
    with patch("rag_agent.api.app.retrieve_chunks", side_effect=OperationalError("x", {}, None)):
        response = client.post(
            "/query",
            headers=_auth_header(token),
            json={"workspace_id": workspace_id, "query": "hello"},
        )
    assert response.status_code == 503


def test_reindex_failure_path(client: TestClient):
    token = _register_and_login(client, "user-z@example.com")
    workspace_id = _create_workspace(client, token, "Org Z", "Workspace Z")
    upload = client.post(
        f"/workspaces/{workspace_id}/documents",
        headers=_auth_header(token),
        files={"file": ("z.md", io.BytesIO(b"hello world"), "text/markdown")},
    )
    doc_id = upload.json()["document_id"]
    with patch(
        "rag_agent.services.embed_text", side_effect=RuntimeError("embedding provider failure")
    ):
        response = client.post(f"/documents/{doc_id}/reindex", headers=_auth_header(token))
    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_query_token_limit(client: TestClient):
    token = _register_and_login(client, "user-h@example.com")
    workspace_id = _create_workspace(client, token, "Org H", "Workspace H")
    response = client.post(
        "/query",
        headers=_auth_header(token),
        json={"workspace_id": workspace_id, "query": "x " * 600},
    )
    assert response.status_code == 413


def test_rate_limit(client: TestClient):
    last = None
    for _ in range(130):
        last = client.get("/health")
        if last.status_code == 429:
            break
    assert last is not None
    assert last.status_code == 429

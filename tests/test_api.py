"""Tests for FastAPI endpoints and REST API contracts."""

import io
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "total_vectors" in data


def test_ui_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "Universal RAG" in res.text


def test_ingest_and_query_flow():
    # 1. Ingest text document
    ingest_res = client.post(
        "/documents",
        json={"source": "Universal RAG supports PDF, CSV, web URLs, and YouTube transcripts with citations."},
    )
    assert ingest_res.status_code == 201
    doc_data = ingest_res.json()
    doc_id = doc_data["id"]
    assert doc_id is not None
    assert doc_data["chunk_count"] >= 1

    # 2. List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["id"] == doc_id for d in docs)

    # 3. Get document details
    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 200
    doc_detail = get_res.json()
    assert len(doc_detail["chunks"]) >= 1

    # 4. Query RAG
    query_res = client.post("/query", json={"question": "What formats does Universal RAG support?"})
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert len(q_data["sources"]) >= 1
    assert "latency_ms" in q_data

    # 5. Delete document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify deletion
    verify_res = client.get(f"/documents/{doc_id}")
    assert verify_res.status_code == 404


def test_upload_file_endpoint():
    file_content = b"Col1,Col2\nVal1,Val2\nVal3,Val4\n"
    res = client.post(
        "/documents/upload",
        files={"file": ("test_upload.csv", io.BytesIO(file_content), "text/csv")},
    )
    assert res.status_code == 201
    data = res.json()
    doc_id = data["id"]

    # Clean up
    client.delete(f"/documents/{doc_id}")


def test_legacy_routes_backward_compatibility():
    # Legacy /ingest-text
    res_text = client.post("/ingest-text?source=Legacy%20text%20ingestion%20content")
    assert res_text.status_code == 200
    doc_id = res_text.json()["id"]

    # Legacy /chat
    res_chat = client.post("/chat", json={"question": "What is legacy text?"})
    assert res_chat.status_code == 200
    assert "answer" in res_chat.json()

    # Legacy /remove-source
    res_rm = client.post("/remove-source", json={"source_id": doc_id})
    assert res_rm.status_code == 200
    assert res_rm.json()["status"] == "removed"


def test_invalid_query_validation():
    res = client.post("/query", json={"question": "   "})
    assert res.status_code == 400


def test_invalid_file_extension():
    res = client.post(
        "/documents/upload",
        files={"file": ("malicious.exe", io.BytesIO(b"binary content"), "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]

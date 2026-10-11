"""Lightweight Persistent Metadata Database using SQLite."""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.config import settings

_local = threading.local()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode enabled."""
    path = db_path or settings.DB_PATH
    conn_key = f"conn_{path}"
    if not hasattr(_local, conn_key):
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        setattr(_local, conn_key, conn)
    return getattr(_local, conn_key)


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                file_size_bytes INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'indexed',
                metadata_json TEXT DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                metadata_json TEXT DEFAULT '{}',
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
            ON chunks(document_id)
            """
        )


def save_document(
    doc_id: str,
    source_name: str,
    source_type: str,
    source_uri: str,
    chunk_count: int = 0,
    file_size_bytes: int = 0,
    status: str = "indexed",
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist or update document metadata."""
    conn = get_connection(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata or {})
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (id, source_name, source_type, source_uri, chunk_count, file_size_bytes, created_at, status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                source_name,
                source_type,
                source_uri,
                chunk_count,
                file_size_bytes,
                created_at,
                status,
                meta_json,
            ),
        )
    return get_document(doc_id, db_path=db_path)


def get_document(doc_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve document metadata by document ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    if not row:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
    return d


def list_documents(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all ingested documents ordered by creation time descending."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        results.append(d)
    return results


def delete_document(doc_id: str, db_path: Optional[str] = None) -> List[str]:
    """
    Delete a document and all its chunks from the database.
    Returns the list of deleted chunk IDs to support vector store synchronization.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
    chunk_ids = [row[0] for row in cursor.fetchall()]

    with conn:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    return chunk_ids


def save_chunks(
    chunks_data: List[Dict[str, Any]],
    db_path: Optional[str] = None,
) -> None:
    """Batch persist chunk metadata into SQLite."""
    conn = get_connection(db_path)
    rows = []
    for item in chunks_data:
        meta_json = json.dumps(item.get("metadata", {}))
        rows.append(
            (
                item["id"],
                item["document_id"],
                item["chunk_index"],
                item["content"],
                item.get("page_number"),
                meta_json,
            )
        )
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (id, document_id, chunk_index, content, page_number, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def get_chunks_for_document(
    doc_id: str, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve all chunks belonging to a document."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
        (doc_id,),
    )
    results = []
    for row in cursor.fetchall():
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        results.append(d)
    return results


def get_all_chunks(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all chunks across all documents."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index ASC")
    results = []
    for row in cursor.fetchall():
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        results.append(d)
    return results


def get_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return summary statistics of stored documents and chunks."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
    }

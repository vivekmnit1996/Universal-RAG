"""Tests for SQLite database persistence."""

import os
from backend.database import (
    delete_document,
    get_all_chunks,
    get_chunks_for_document,
    get_document,
    get_stats,
    init_db,
    list_documents,
    save_chunks,
    save_document,
)


def test_database_crud_operations(temp_dir):
    db_path = os.path.join(temp_dir, "test.db")
    init_db(db_path)

    # 1. Save document
    doc = save_document(
        doc_id="d1",
        source_name="ml_notes.pdf",
        source_type="pdf",
        source_uri="/tmp/ml_notes.pdf",
        chunk_count=2,
        file_size_bytes=1024,
        metadata={"author": "Researcher"},
        db_path=db_path,
    )
    assert doc is not None
    assert doc["id"] == "d1"
    assert doc["source_name"] == "ml_notes.pdf"
    assert doc["metadata"]["author"] == "Researcher"

    # 2. Save chunks
    chunks_data = [
        {
            "id": "d1#0",
            "document_id": "d1",
            "chunk_index": 0,
            "content": "Intro to deep learning.",
            "page_number": 1,
            "metadata": {"source_name": "ml_notes.pdf"},
        },
        {
            "id": "d1#1",
            "document_id": "d1",
            "chunk_index": 1,
            "content": "Transformers and self-attention.",
            "page_number": 2,
            "metadata": {"source_name": "ml_notes.pdf"},
        },
    ]
    save_chunks(chunks_data, db_path=db_path)

    # 3. Retrieve chunks
    doc_chunks = get_chunks_for_document("d1", db_path=db_path)
    assert len(doc_chunks) == 2
    assert doc_chunks[0]["chunk_index"] == 0
    assert doc_chunks[1]["page_number"] == 2

    # 4. List documents
    docs = list_documents(db_path=db_path)
    assert len(docs) == 1
    assert docs[0]["id"] == "d1"

    # 5. Check stats
    stats = get_stats(db_path=db_path)
    assert stats["total_documents"] == 1
    assert stats["total_chunks"] == 2

    # 6. Delete document
    deleted_chunk_ids = delete_document("d1", db_path=db_path)
    assert set(deleted_chunk_ids) == {"d1#0", "d1#1"}
    assert get_document("d1", db_path=db_path) is None
    assert len(get_chunks_for_document("d1", db_path=db_path)) == 0

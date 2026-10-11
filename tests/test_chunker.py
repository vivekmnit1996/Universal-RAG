"""Tests for chunking engine and metadata preservation."""

from backend.chunker import chunk_document_batch
from backend.document_parser import ParsedDocument


def test_chunking_metadata_preservation():
    parsed = [
        ParsedDocument(
            content="Alpha section content for chunking. " * 30,
            metadata={
                "source_name": "guide.pdf",
                "source_type": "pdf",
                "source_uri": "/path/to/guide.pdf",
                "page_number": 2,
            },
        )
    ]

    chunks = chunk_document_batch(parsed, document_id="doc_123", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1

    for idx, c in enumerate(chunks):
        assert c.metadata["document_id"] == "doc_123"
        assert c.metadata["chunk_id"] == f"doc_123#{idx}"
        assert c.metadata["chunk_index"] == idx
        assert c.metadata["source_name"] == "guide.pdf"
        assert c.metadata["source_type"] == "pdf"
        assert c.metadata["page_number"] == 2
        assert c.metadata["total_chunks"] == len(chunks)


def test_chunking_size_and_overlap():
    text = "Short text under chunk size."
    parsed = [
        ParsedDocument(
            content=text,
            metadata={"source_name": "short.txt", "source_type": "text"},
        )
    ]

    chunks = chunk_document_batch(parsed, document_id="doc_456", chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == text


def test_chunking_empty_content():
    parsed = [
        ParsedDocument(content="", metadata={"source_name": "empty.txt"}),
        ParsedDocument(content="   ", metadata={"source_name": "spaces.txt"}),
    ]
    chunks = chunk_document_batch(parsed, document_id="doc_empty")
    assert len(chunks) == 0

"""Tests for source citations extraction and formatting."""

from langchain_core.documents import Document
from backend.citations import build_citations, format_citations_text


def test_build_citations_multi_source():
    retrieved = [
        (
            Document(
                page_content="Deep learning algorithms use multiple layers of representations.",
                metadata={
                    "source_name": "ai_handbook.pdf",
                    "source_type": "pdf",
                    "page_number": 4,
                    "chunk_id": "doc1#0",
                    "source_uri": "/docs/ai_handbook.pdf",
                },
            ),
            0.88,
        ),
        (
            Document(
                page_content="Row 1: Model=ResNet, Accuracy=0.92",
                metadata={
                    "source_name": "models.csv",
                    "source_type": "csv",
                    "row_start": 1,
                    "row_end": 20,
                    "chunk_id": "doc2#0",
                    "source_uri": "/data/models.csv",
                },
            ),
            0.79,
        ),
        (
            Document(
                page_content="Video transcript discussing transformer architectures.",
                metadata={
                    "source_name": "Transformers Explained",
                    "source_type": "youtube",
                    "video_id": "abcdef12345",
                    "chunk_id": "doc3#0",
                    "source_uri": "https://youtube.com/watch?v=abcdef12345",
                },
            ),
            0.75,
        ),
        (
            Document(
                page_content="Web article on retrieval augmented generation techniques.",
                metadata={
                    "source_name": "https://example.com/rag",
                    "source_type": "web",
                    "title": "Modern RAG Patterns",
                    "chunk_id": "doc4#0",
                    "source_uri": "https://example.com/rag",
                },
            ),
            0.82,
        ),
    ]

    citations = build_citations(retrieved)
    assert len(citations) == 4

    assert citations[0]["label"] == "ai_handbook.pdf — Page 4"
    assert citations[0]["page"] == 4

    assert citations[1]["label"] == "models.csv — Rows 1-20"

    assert citations[2]["label"] == "YouTube (abcdef12345) — Video Transcript"

    assert citations[3]["label"] == "Modern RAG Patterns — Web Source"

    text = format_citations_text(citations)
    assert "ai_handbook.pdf — Page 4" in text
    assert "models.csv — Rows 1-20" in text


def test_build_citations_deduplication():
    # Two chunks from the same page of the same PDF
    retrieved = [
        (
            Document(
                page_content="Section 1 of page 3",
                metadata={"source_name": "manual.pdf", "source_type": "pdf", "page_number": 3, "chunk_id": "c1"},
            ),
            0.9,
        ),
        (
            Document(
                page_content="Section 2 of page 3",
                metadata={"source_name": "manual.pdf", "source_type": "pdf", "page_number": 3, "chunk_id": "c2"},
            ),
            0.85,
        ),
    ]

    citations = build_citations(retrieved)
    assert len(citations) == 1
    assert citations[0]["label"] == "manual.pdf — Page 3"

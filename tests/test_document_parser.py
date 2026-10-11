"""Tests for document parsers across multiple sources."""

import os
from unittest.mock import MagicMock, patch
import pytest

from backend.document_parser import (
    DocumentParseError,
    detect_source_type,
    extract_youtube_video_id,
    parse_csv,
    parse_pdf,
    parse_source,
    parse_text,
    parse_web,
    parse_youtube,
)
from backend.security import ValidationError


def test_detect_source_type():
    assert detect_source_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"
    assert detect_source_type("https://youtu.be/dQw4w9WgXcQ") == "youtube"
    assert detect_source_type("https://example.com/article") == "web"
    assert detect_source_type("report.pdf") == "pdf"
    assert detect_source_type("dataset.csv") == "csv"
    assert detect_source_type("notes.txt") == "text"


def test_parse_text(sample_text_path):
    docs = parse_text(sample_text_path)
    assert len(docs) == 1
    assert "LangGraph" in docs[0].content
    assert docs[0].metadata["source_type"] == "text"


def test_parse_raw_text_string():
    docs = parse_text("Universal RAG handles diverse formats.")
    assert len(docs) == 1
    assert "Universal RAG" in docs[0].content


def test_parse_csv(sample_csv_path):
    docs = parse_csv(sample_csv_path)
    assert len(docs) >= 1
    assert "ResNet" in docs[0].content
    assert "BERT" in docs[0].content
    assert docs[0].metadata["source_type"] == "csv"
    assert docs[0].metadata["total_rows"] == 3


def test_parse_empty_csv(temp_dir):
    empty_csv = os.path.join(temp_dir, "empty.csv")
    with open(empty_csv, "w") as f:
        f.write("")
    with pytest.raises(DocumentParseError):
        parse_csv(empty_csv)


def test_parse_web():
    mock_resp = MagicMock()
    mock_resp.text = """
    <html>
      <head><title>RAG Overview Article</title></head>
      <body>
        <h1>Understanding RAG Architecture</h1>
        <p>Retrieval-Augmented Generation improves LLM reliability by grounding responses in private documents.</p>
      </body>
    </html>
    """
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        docs = parse_web("https://example.com/rag-guide")
        assert len(docs) == 1
        assert "Retrieval-Augmented Generation" in docs[0].content
        assert docs[0].metadata["source_type"] == "web"
        assert docs[0].metadata["title"] == "RAG Overview Article"


def test_extract_youtube_id():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=YdOXS_9_P4U") == "YdOXS_9_P4U"
    assert extract_youtube_video_id("https://youtu.be/YdOXS_9_P4U") == "YdOXS_9_P4U"
    assert extract_youtube_video_id("https://www.youtube.com/embed/YdOXS_9_P4U") == "YdOXS_9_P4U"


def test_parse_youtube():
    mock_transcript = [
        {"text": "Welcome to the tutorial on vector embeddings.", "start": 0.0, "duration": 2.0},
        {"text": "We will explore FAISS and LangChain indexing.", "start": 2.0, "duration": 3.0},
    ]

    with patch("youtube_transcript_api.YouTubeTranscriptApi.fetch", return_value=mock_transcript):
        docs = parse_youtube("https://www.youtube.com/watch?v=YdOXS_9_P4U")
        assert len(docs) == 1
        assert "vector embeddings" in docs[0].content
        assert docs[0].metadata["source_type"] == "youtube"
        assert docs[0].metadata["video_id"] == "YdOXS_9_P4U"


def test_parse_invalid_file_handling():
    with pytest.raises(DocumentParseError):
        parse_pdf("/nonexistent/path/to/missing.pdf")


def test_parse_invalid_url_handling():
    with pytest.raises(ValidationError):
        parse_web("ftp://unsupported-protocol.com")
    with pytest.raises(ValidationError):
        parse_web("http://localhost/secret")

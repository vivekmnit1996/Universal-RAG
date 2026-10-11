"""Pytest fixtures for Universal RAG tests."""

import os
import shutil
import tempfile
import pytest
from pypdf import PdfWriter

from backend.config import Settings
from backend.database import init_db
from backend.vector_store import VectorStoreManager
from backend.service import RAGService


@pytest.fixture
def temp_dir():
    """Create a temporary directory for isolated test execution."""
    td = tempfile.mkdtemp(prefix="rag_test_")
    yield td
    shutil.rmtree(td, ignore_errors=True)


@pytest.fixture
def isolated_service(temp_dir):
    """Create an isolated instance of RAGService using temporary database and vector store."""
    test_db = os.path.join(temp_dir, "test_meta.db")
    test_vs = os.path.join(temp_dir, "test_vectorstore")
    test_up = os.path.join(temp_dir, "test_uploads")
    os.makedirs(test_vs, exist_ok=True)
    os.makedirs(test_up, exist_ok=True)

    init_db(test_db)

    svc = RAGService.__new__(RAGService)
    svc.vector_manager = VectorStoreManager(persist_dir=test_vs)
    from backend.rag_workflow import RAGWorkflow
    svc.workflow = RAGWorkflow(svc.vector_manager)
    return svc


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Generate a valid multi-page PDF for test ingestion."""
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # We can write text to page or use writer with an existing stream
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


@pytest.fixture
def sample_csv_path(temp_dir):
    """Generate a sample CSV file."""
    csv_path = os.path.join(temp_dir, "sample.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,model,accuracy\n1,ResNet,0.92\n2,BERT,0.94\n3,GPT-4,0.98\n")
    return csv_path


@pytest.fixture
def sample_text_path(temp_dir):
    """Generate a sample plain text file."""
    txt_path = os.path.join(temp_dir, "sample.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It coordinates cyclic workflows and conditional branching."
        )
    return txt_path


@pytest.fixture(autouse=True)
def clean_session_memory():
    """Ensure session memory is cleared before and after each test."""
    from backend.session_memory import session_memory
    session_memory.clear_all()
    yield
    session_memory.clear_all()


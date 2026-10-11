"""Tests for LangGraph RAG stateful workflow."""

import os
from unittest.mock import MagicMock
from langchain_core.documents import Document
from backend.rag_workflow import RAGWorkflow
from backend.vector_store import VectorStoreManager


def test_workflow_execution_with_context(temp_dir):
    vs_dir = os.path.join(temp_dir, "wf_vs")
    vm = VectorStoreManager(persist_dir=vs_dir)

    # Ingest document chunk
    doc = Document(
        page_content="Vector databases index embeddings for high-speed approximate nearest neighbor search.",
        metadata={
            "source_name": "vector_db.pdf",
            "source_type": "pdf",
            "page_number": 1,
            "chunk_id": "vdb#0",
        },
    )
    vm.add_chunks([doc])

    workflow = RAGWorkflow(vector_manager=vm)
    result = workflow.run("How do vector databases index embeddings?")

    assert result["question"] == "How do vector databases index embeddings?"
    assert len(result["citations"]) >= 1
    assert "vector_db.pdf" in result["citations"][0]["source_name"]
    assert "vector databases" in result["answer"].lower() or "approximate nearest neighbor" in result["answer"].lower()


def test_workflow_execution_no_context(temp_dir):
    vs_dir = os.path.join(temp_dir, "empty_vs")
    vm = VectorStoreManager(persist_dir=vs_dir)

    workflow = RAGWorkflow(vector_manager=vm)
    result = workflow.run("What is quantum computing?")

    assert len(result["citations"]) == 0
    assert "could not find relevant information" in result["answer"].lower()


def test_workflow_with_mock_llm(temp_dir):
    vs_dir = os.path.join(temp_dir, "mock_llm_vs")
    vm = VectorStoreManager(persist_dir=vs_dir)

    doc = Document(
        page_content="Universal RAG provides grounded answers with source citations.",
        metadata={"source_name": "rag.txt", "source_type": "text", "chunk_id": "r#0"},
    )
    vm.add_chunks([doc])

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Universal RAG uses document chunks to provide verifiable answers with citations."
    )

    workflow = RAGWorkflow(vector_manager=vm, llm=mock_llm)
    result = workflow.run("What does Universal RAG provide?")

    assert mock_llm.invoke.called
    assert "Universal RAG" in result["answer"]
    assert len(result["citations"]) == 1

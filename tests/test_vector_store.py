"""Tests for FAISS persistent vector store and incremental operations."""

import os
from langchain_core.documents import Document
from backend.vector_store import VectorStoreManager


def test_incremental_indexing_and_persistence(temp_dir):
    vs_dir = os.path.join(temp_dir, "vs_store")
    vm = VectorStoreManager(persist_dir=vs_dir)

    # 1. Add first batch of chunks
    batch1 = [
        Document(
            page_content="Retrieval-Augmented Generation combines retrieval with generative LLMs.",
            metadata={"chunk_id": "c1", "document_id": "doc_a", "source_name": "rag.pdf"},
        ),
        Document(
            page_content="Vector embeddings capture semantic distance in high-dimensional space.",
            metadata={"chunk_id": "c2", "document_id": "doc_a", "source_name": "rag.pdf"},
        ),
    ]
    added = vm.add_chunks(batch1)
    assert added == 2
    assert vm.get_total_vectors() == 2

    # 2. Add second batch incrementally without re-embedding batch1
    batch2 = [
        Document(
            page_content="LangGraph orchestrates stateful workflow graphs with cyclic execution.",
            metadata={"chunk_id": "c3", "document_id": "doc_b", "source_name": "langgraph.md"},
        )
    ]
    vm.add_chunks(batch2)
    assert vm.get_total_vectors() == 3

    # 3. Test persistence across instance reloads
    vm_reloaded = VectorStoreManager(persist_dir=vs_dir)
    assert vm_reloaded.get_total_vectors() == 3

    # Search for batch 1 topic
    results_rag = vm_reloaded.similarity_search_with_scores("Retrieval-Augmented Generation", k=2, threshold=0.0)
    assert len(results_rag) >= 1
    assert any("Retrieval-Augmented Generation" in doc.page_content for doc, _ in results_rag)

    # Search for batch 2 topic
    results_graph = vm_reloaded.similarity_search_with_scores("LangGraph stateful workflow", k=2, threshold=0.0)
    assert len(results_graph) >= 1
    assert any("LangGraph" in doc.page_content for doc, _ in results_graph)

    # 4. Incremental deletion of doc_a chunks
    deleted_count = vm_reloaded.delete_chunks(["c1", "c2"])
    assert deleted_count == 2
    assert vm_reloaded.get_total_vectors() == 1

    # Verify only doc_b remains
    remaining_res = vm_reloaded.similarity_search_with_scores("workflow", k=2, threshold=0.0)
    assert len(remaining_res) == 1
    assert remaining_res[0][0].metadata["chunk_id"] == "c3"


def test_similarity_threshold_filtering(temp_dir):
    vs_dir = os.path.join(temp_dir, "vs_thresh")
    vm = VectorStoreManager(persist_dir=vs_dir)

    docs = [
        Document(
            page_content="Deep neural networks require backpropagation for gradient descent optimization.",
            metadata={"chunk_id": "nn_1"},
        )
    ]
    vm.add_chunks(docs)

    # Search with high threshold on unrelated query
    results = vm.similarity_search_with_scores("baking chocolate chip cookies in an oven", k=1, threshold=0.85)
    assert len(results) == 0


def test_duplicate_chunk_deduplication(temp_dir):
    vs_dir = os.path.join(temp_dir, "vs_dedup")
    vm = VectorStoreManager(persist_dir=vs_dir)

    duplicate_text = "This is a duplicate sentence repeated across multiple pages in the document."
    docs = [
        Document(page_content=duplicate_text, metadata={"chunk_id": "dup_1"}),
        Document(page_content=duplicate_text, metadata={"chunk_id": "dup_2"}),
    ]
    vm.add_chunks(docs)

    results = vm.similarity_search_with_scores("duplicate sentence", k=2, threshold=0.0)
    # Deduplication should collapse duplicate content to 1 result
    assert len(results) == 1

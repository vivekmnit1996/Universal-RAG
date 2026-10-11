"""Tests for conversational memory, intent classification, session isolation, and RAG separation."""

import os
from unittest.mock import MagicMock
from langchain_core.documents import Document
from backend.rag_workflow import RAGWorkflow, is_casual_query
from backend.session_memory import session_memory
from backend.vector_store import VectorStoreManager
from fastapi.testclient import TestClient
from app import app


def test_casual_intent_classification():
    assert is_casual_query("hii")[0] is True
    assert is_casual_query("hello")[0] is True
    assert is_casual_query("hey there")[0] is True
    assert is_casual_query("thanks!")[0] is True
    assert is_casual_query("thank you so much")[0] is True
    assert is_casual_query("bye")[0] is True
    assert is_casual_query("good morning")[0] is True

    # Knowledge queries should be classified as knowledge
    assert is_casual_query("What is Retrieval-Augmented Generation?")[0] is False
    assert is_casual_query("How does FAISS index embeddings?")[0] is False
    assert is_casual_query("Tell me about normalization in deep learning")[0] is False


def test_greeting_no_vector_retrieval(temp_dir):
    vs_dir = os.path.join(temp_dir, "test_vs_greeting")
    vm = VectorStoreManager(persist_dir=vs_dir)

    # Ingest a sample knowledge document
    doc = Document(
        page_content="Vector databases index dense embeddings for similarity search.",
        metadata={"source_name": "vector_guide.pdf", "chunk_id": "v1"},
    )
    vm.add_chunks([doc])

    workflow = RAGWorkflow(vector_manager=vm)
    result = workflow.run("hii", session_id="test_sess_greeting")

    assert result["intent"] == "casual"
    assert "hello" in result["answer"].lower() or "universal rag" in result["answer"].lower()
    # MUST have zero sources and zero retrieved chunks
    assert result["citations"] == []
    assert result["sources_count"] == 0


def test_irrelevant_query_fallback_and_no_citations(temp_dir):
    vs_dir = os.path.join(temp_dir, "test_vs_fallback")
    vm = VectorStoreManager(persist_dir=vs_dir)

    doc = Document(
        page_content="Photosynthesis is the process by which green plants make food.",
        metadata={"source_name": "biology.pdf", "chunk_id": "b1"},
    )
    vm.add_chunks([doc])

    workflow = RAGWorkflow(vector_manager=vm)
    # Ask completely unrelated knowledge question with threshold enforcement
    result = workflow.run("How do quantum supercomputers factor prime integers?", session_id="test_sess_fallback")

    # Should gracefully indicate lack of information and have ZERO citations
    ans_lower = result["answer"].lower()
    assert any(phrase in ans_lower for phrase in ["couldn't find", "could not find", "not present in the sources", "not available"])
    assert result["citations"] == []


def test_session_isolation(temp_dir):
    vs_dir = os.path.join(temp_dir, "test_vs_isolation")
    vm = VectorStoreManager(persist_dir=vs_dir)

    workflow = RAGWorkflow(vector_manager=vm)

    sess_a = "session_alice"
    sess_b = "session_bob"

    session_memory.clear_session(sess_a)
    session_memory.clear_session(sess_b)

    # Alice asks a question
    workflow.run("What is machine learning?", session_id=sess_a)

    # Bob asks a question
    workflow.run("What is quantum mechanics?", session_id=sess_b)

    hist_a = session_memory.get_history(sess_a)
    hist_b = session_memory.get_history(sess_b)

    assert len(hist_a) == 2  # HumanMessage + AIMessage
    assert len(hist_b) == 2  # HumanMessage + AIMessage

    assert "machine learning" in hist_a[0].content.lower()
    assert "quantum mechanics" in hist_b[0].content.lower()

    # Verify Alice's history has no trace of Bob's question
    assert not any("quantum" in m.content.lower() for m in hist_a)
    # Verify Bob's history has no trace of Alice's question
    assert not any("machine learning" in m.content.lower() for m in hist_b)


def test_vector_store_never_indexes_chat_turns():
    client = TestClient(app)

    # 1. Ingest an explicit document
    ingest_res = client.post(
        "/documents",
        json={"source": "The capital of France is Paris. Paris is known for the Eiffel Tower."},
    )
    assert ingest_res.status_code == 201
    doc_id = ingest_res.json()["id"]

    from backend.service import service
    vectors_before = service.vector_manager.get_total_vectors()

    # 2. Execute multiple chat queries in a session
    client.post("/query", json={"question": "hii", "session_id": "chat_isolation_test"})
    client.post("/query", json={"question": "What is the capital of France?", "session_id": "chat_isolation_test"})
    client.post("/query", json={"question": "thanks!", "session_id": "chat_isolation_test"})

    # 3. Check vector count - MUST REMAIN UNCHANGED
    vectors_after = service.vector_manager.get_total_vectors()
    assert vectors_before == vectors_after, "Vector store must NEVER index chat queries or assistant responses!"

    # Clean up
    client.delete(f"/documents/{doc_id}")


def test_clear_session_endpoint():
    client = TestClient(app)
    sess_id = "test_clear_sess_123"

    # Send a query to establish session history
    client.post("/query", json={"question": "hii", "session_id": sess_id})

    # Verify history exists
    hist_res = client.get(f"/sessions/{sess_id}/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()["messages"]) >= 2

    # Clear session
    clear_res = client.post(f"/sessions/{sess_id}/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["status"] == "cleared"

    # Verify history is now empty
    hist_after = client.get(f"/sessions/{sess_id}/history")
    assert len(hist_after.json()["messages"]) == 0


def test_list_and_delete_sessions():
    client = TestClient(app)
    sid = "sess_list_test_99"

    # Create turn
    client.post("/query", json={"question": "Who is the candidate in the document?", "session_id": sid})

    # List sessions
    list_res = client.get("/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert any(s["session_id"] == sid for s in sessions)
    matching = next(s for s in sessions if s["session_id"] == sid)
    assert "candidate" in matching["title"].lower()

    # Delete session
    del_res = client.delete(f"/sessions/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify deleted
    list_res_after = client.get("/sessions")
    assert not any(s["session_id"] == sid for s in list_res_after.json())


def test_multi_turn_context_continuity(temp_dir):
    """Verify that multi-turn pronouns (he/his) and conversational cues resolve context without strict fallback."""
    vs_dir = os.path.join(temp_dir, "test_vs_multiturn")
    vm = VectorStoreManager(persist_dir=vs_dir)

    doc = Document(
        page_content="Vivek is a senior software engineer who built automated payment routing at Juspay using Python and FastAPI.",
        metadata={"source_name": "resume.pdf", "chunk_id": "c1"},
    )
    vm.add_chunks([doc])

    workflow = RAGWorkflow(vector_manager=vm)
    sess_id = "test_continuity_sess"

    # Turn 1: Initial query
    res1 = workflow.run("Who is Vivek?", session_id=sess_id)
    assert "Vivek" in res1["answer"] or "engineer" in res1["answer"]

    # Turn 2: Follow-up using 'he' (pronoun resolution)
    res2 = workflow.run("What did he build?", session_id=sess_id)
    # Standalone query should resolve context
    assert "Vivek" in res2.get("standalone_query", "") or "payment" in res2["answer"].lower() or "juspay" in res2["answer"].lower()
    assert "I could not find relevant information" not in res2["answer"]

    # Turn 3: Evaluation query (not strictly refused)
    res3 = workflow.run("Evaluate his experience for a backend role", session_id=sess_id)
    assert "I could not find relevant information" not in res3["answer"]

"""Unified Application Service Layer orchestrating ingestion, retrieval, and persistence."""

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend import database as db
from backend.chunker import chunk_document_batch
from backend.config import settings
from backend.document_parser import (
    DocumentParseError,
    detect_source_type,
    parse_source,
)
from backend.rag_workflow import RAGWorkflow
from backend.security import (
    ValidationError,
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
    validate_url,
)
from backend.vector_store import VectorStoreManager


class RAGService:
    """Singleton service coordinator for Universal RAG operations."""

    def __init__(self):
        settings.ensure_directories()
        db.init_db(settings.DB_PATH)
        self.vector_manager = VectorStoreManager()
        self.workflow = RAGWorkflow(self.vector_manager)

    def ingest_file(self, temp_file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Ingest an uploaded file incrementally:
        1. Sanitize & validate filename and size
        2. Parse file pages / rows
        3. Chunk into metadata-enriched chunks
        4. Persist document & chunk metadata in SQLite
        5. Embed and add only new chunks to FAISS vector store
        """
        clean_name = sanitize_filename(original_filename)
        validate_file_extension(clean_name)

        file_size = os.path.getsize(temp_file_path)
        validate_file_size(file_size)

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(clean_name)[1]
        permanent_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}{ext}")
        shutil.move(temp_file_path, permanent_path)

        # Parse document
        parsed_docs = parse_source(permanent_path, source_name=clean_name)
        if not parsed_docs:
            raise DocumentParseError(f"No content extracted from {clean_name}")

        source_type = parsed_docs[0].metadata.get("source_type", "file")

        # Chunk document
        chunks = chunk_document_batch(parsed_docs, document_id=doc_id)
        chunk_count = len(chunks)

        # Save to SQLite
        db.save_document(
            doc_id=doc_id,
            source_name=clean_name,
            source_type=source_type,
            source_uri=permanent_path,
            chunk_count=chunk_count,
            file_size_bytes=file_size,
            status="indexed",
            metadata={"original_name": original_filename},
        )

        chunk_records = [
            {
                "id": c.metadata["chunk_id"],
                "document_id": doc_id,
                "chunk_index": c.metadata["chunk_index"],
                "content": c.page_content,
                "page_number": c.metadata.get("page_number"),
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        db.save_chunks(chunk_records)

        # Incrementally index vectors
        self.vector_manager.add_chunks(chunks)

        return db.get_document(doc_id)

    def ingest_text_or_url(self, source_input: str) -> Dict[str, Any]:
        """
        Ingest a URL (web or YouTube) or raw text snippet incrementally.
        """
        source_clean = source_input.strip()
        if not source_clean:
            raise ValidationError("Input text or URL cannot be empty")

        source_type = detect_source_type(source_clean)
        if source_type in ("web", "youtube"):
            validate_url(source_clean)

        doc_id = str(uuid.uuid4())
        parsed_docs = parse_source(source_clean)
        if not parsed_docs:
            raise DocumentParseError(f"No content extracted from source: {source_clean[:50]}")

        source_name = parsed_docs[0].metadata.get("source_name", source_clean[:50])

        chunks = chunk_document_batch(parsed_docs, document_id=doc_id)
        chunk_count = len(chunks)

        db.save_document(
            doc_id=doc_id,
            source_name=source_name,
            source_type=source_type,
            source_uri=source_clean,
            chunk_count=chunk_count,
            file_size_bytes=len(source_clean.encode("utf-8")),
            status="indexed",
            metadata={"raw_input_snippet": source_clean[:100]},
        )

        chunk_records = [
            {
                "id": c.metadata["chunk_id"],
                "document_id": doc_id,
                "chunk_index": c.metadata["chunk_index"],
                "content": c.page_content,
                "page_number": c.metadata.get("page_number"),
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        db.save_chunks(chunk_records)

        # Incrementally index vectors
        self.vector_manager.add_chunks(chunks)

        return db.get_document(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents with status and metadata."""
        return db.list_documents()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata and chunk list."""
        doc = db.get_document(doc_id)
        if not doc:
            return None
        doc["chunks"] = db.get_chunks_for_document(doc_id)
        return doc

    def delete_document(self, doc_id: str) -> bool:
        """
        Incrementally delete a document from SQLite and FAISS vector index.
        Removes associated uploaded physical file if present.
        """
        doc = db.get_document(doc_id)
        if not doc:
            return False

        # Delete from DB and get deleted chunk IDs
        deleted_chunk_ids = db.delete_document(doc_id)

        # Incrementally remove vectors from FAISS
        self.vector_manager.delete_chunks(deleted_chunk_ids)

        # Remove physical file if in UPLOAD_DIR
        file_path = doc.get("source_uri")
        if file_path and os.path.exists(file_path) and settings.UPLOAD_DIR in file_path:
            try:
                os.remove(file_path)
            except OSError:
                pass

        return True

    def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute conversational grounded question answering through LangGraph workflow."""
        q = question.strip()
        if not q:
            raise ValidationError("Question cannot be empty")

        sess_id = session_id or "default"
        from backend.rag_workflow import is_casual_query
        is_casual, _ = is_casual_query(q)

        total_vectors = self.vector_manager.get_total_vectors()
        if total_vectors == 0 and not is_casual:
            return {
                "question": q,
                "session_id": sess_id,
                "intent": "knowledge",
                "answer": "No documents have been ingested yet. Please upload files or links first.",
                "sources": [],
                "standalone_query": q,
                "latency_ms": 0,
            }

        start_time = time.time()
        result = self.workflow.run(q, session_id=sess_id)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "question": result["question"],
            "session_id": result.get("session_id", sess_id),
            "intent": result.get("intent", "knowledge"),
            "answer": result["answer"],
            "sources": result["citations"],
            "standalone_query": result.get("standalone_query"),
            "latency_ms": latency_ms,
        }

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        from backend.session_memory import session_memory
        return session_memory.clear_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session completely."""
        from backend.session_memory import session_memory
        return session_memory.delete_session(session_id)

    def clear_all_sessions(self) -> None:
        """Clear all conversation sessions."""
        from backend.session_memory import session_memory
        session_memory.clear_all()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List active conversation sessions."""
        from backend.session_memory import session_memory
        return session_memory.list_sessions()

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve conversation history for a specific session."""
        from backend.session_memory import session_memory
        return session_memory.get_session_messages(session_id)

    def evaluate_resume(
        self,
        document_id: Optional[str] = None,
        job_description: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate an uploaded resume against the 100-point rubric.
        Supports automatic resume detection, explicit document_id, or direct text.
        """
        from backend.resume_evaluator import evaluate_resume as run_resume_eval

        target_text = ""
        doc_name = "Direct Text Submission"
        doc_id = document_id

        if raw_text and raw_text.strip():
            target_text = raw_text.strip()
        elif document_id:
            doc = db.get_document(document_id)
            if not doc:
                raise ValidationError(f"Document with ID '{document_id}' not found.")
            doc_name = doc.get("source_name", "Uploaded Resume")
            chunks = db.get_chunks_for_document(document_id)
            target_text = "\n\n".join(c["content"] for c in chunks)
            if not target_text and os.path.exists(doc.get("source_uri", "")):
                parsed = parse_source(doc["source_uri"])
                target_text = "\n\n".join(p.content for p in parsed)
        else:
            # Auto-detect uploaded resume document from SQLite
            docs = db.list_documents()
            if not docs:
                raise ValidationError("No documents uploaded yet. Please upload a resume PDF first.")

            resume_doc = None
            # Prioritize documents with 'resume' or 'cv' in name
            for d in docs:
                name_lower = d.get("source_name", "").lower()
                if "resume" in name_lower or "cv" in name_lower:
                    resume_doc = d
                    break
            # Fallback to the latest PDF document, then any latest document
            if not resume_doc:
                pdf_docs = [d for d in docs if d.get("source_type") == "pdf"]
                resume_doc = pdf_docs[0] if pdf_docs else docs[0]

            doc_id = resume_doc["id"]
            doc_name = resume_doc.get("source_name", "Uploaded Resume")
            chunks = db.get_chunks_for_document(doc_id)
            target_text = "\n\n".join(c["content"] for c in chunks)
            if not target_text and os.path.exists(resume_doc.get("source_uri", "")):
                parsed = parse_source(resume_doc["source_uri"])
                target_text = "\n\n".join(p.content for p in parsed)

        if not target_text.strip():
            raise ValidationError(f"No extractable text found in '{doc_name}' to evaluate.")

        res = run_resume_eval(
            resume_text=target_text,
            source_name=doc_name,
            document_id=doc_id,
            job_description=job_description,
        )
        return res.model_dump()

    def get_health(self) -> Dict[str, Any]:
        """Check system status, database stats, and vector store indices."""
        stats = db.get_stats()
        has_api_key = bool(
            settings.OPENAI_API_KEY
            and not settings.OPENAI_API_KEY.startswith("your-")
            and len(settings.OPENAI_API_KEY) > 10
        )
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "total_vectors": self.vector_manager.get_total_vectors(),
            "llm_provider": "OpenAI" if has_api_key else "Offline-Fallback",
            "llm_model": settings.LLM_MODEL if has_api_key else "Local-Grounded",
            "embedding_model": settings.EMBEDDING_MODEL if has_api_key else "Fallback-Deterministic-1536",
        }


# Singleton service instance
service = RAGService()

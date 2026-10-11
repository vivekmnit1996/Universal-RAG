"""Universal RAG FastAPI Application.

Provides clean REST endpoints for document ingestion, persistence management,
health monitoring, and grounded question answering with verifiable citations.
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.document_parser import DocumentParseError
from backend.security import ValidationError
from backend.service import service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("universal_rag")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A multi-source, grounded, scalable RAG system with persistent vector indexing and source citations.",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------
class IngestTextRequest(BaseModel):
    source: str = Field(..., description="Web URL, YouTube URL, or raw text content to ingest")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User query to answer using ingested documents")
    session_id: Optional[str] = Field(None, description="Unique conversation session identifier")


class RemoveSourceRequest(BaseModel):
    source_id: str = Field(..., description="Document ID to remove from database and vector index")


class ResumeEvaluationRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Document ID of uploaded resume")
    job_description: Optional[str] = Field(None, description="Optional job description for skill and keyword tailoring")
    raw_text: Optional[str] = Field(None, description="Optional raw resume text if no document was uploaded")


class CitationItem(BaseModel):
    label: str
    source_name: str
    source_type: str
    source_uri: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[CitationItem]
    session_id: str = "default"
    intent: str = "knowledge"
    standalone_query: Optional[str] = None
    latency_ms: float = 0.0


# -------------------------------------------------------------
# EXCEPTION HANDLERS
# -------------------------------------------------------------
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "ValidationError"},
    )


@app.exception_handler(DocumentParseError)
async def parse_exception_handler(request, exc: DocumentParseError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "DocumentParseError"},
    )


# -------------------------------------------------------------
# WEB UI
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_ui():
    """Serve the web application user interface."""
    ui_path = os.path.join(settings.BASE_DIR, "frontend", "index.html")
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="UI template not found")
    with open(ui_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------------------------------------
# RESTFUL CORE API ENDPOINTS
# -------------------------------------------------------------
@app.get("/health", summary="Health & System Status")
def health_check():
    """Return system health status, storage metrics, and active models."""
    return service.get_health()


@app.get("/documents", summary="List All Documents")
def list_documents():
    """Retrieve all persisted documents with chunk counts and ingestion status."""
    return service.list_documents()


@app.get("/documents/{document_id}", summary="Get Document Details")
def get_document(document_id: str):
    """Retrieve specific document metadata along with all its indexed chunks."""
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return doc


@app.post("/documents", summary="Ingest URL or Text", status_code=status.HTTP_201_CREATED)
def ingest_document(req: IngestTextRequest):
    """Ingest a web URL, YouTube URL, or raw text string into persistent store and vector index."""
    try:
        doc = service.ingest_text_or_url(req.source)
        return doc
    except (ValidationError, DocumentParseError):
        raise
    except Exception as e:
        logger.exception("Failed to ingest text/URL source: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")


@app.post("/documents/upload", summary="Upload & Ingest File", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Upload and incrementally index a PDF, CSV, or text document."""
    temp_fd, temp_path = tempfile.mkstemp()
    try:
        with os.fdopen(temp_fd, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc = service.ingest_file(temp_path, file.filename or "uploaded_file")
        return doc
    except (ValidationError, DocumentParseError):
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.exception("Failed to upload and ingest file: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during file processing")


@app.delete("/documents/{document_id}", summary="Delete Document")
def delete_document(document_id: str):
    """Incrementally remove a document and its vectors from storage."""
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return {"status": "deleted", "document_id": document_id}


@app.post("/query", response_model=QueryResponse, summary="Query RAG System")
def query_rag(req: QueryRequest):
    """
    Execute grounded query answering over indexed documents using LangGraph workflow.
    Supports session-based conversational context while keeping chat history isolated from RAG vector knowledge.
    """
    try:
        return service.query(req.question, session_id=req.session_id)
    except ValidationError:
        raise
    except Exception as e:
        logger.exception("Query execution failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during query processing")


@app.get("/sessions", summary="List All Sessions")
def list_sessions():
    """Retrieve all active conversation sessions and their metadata."""
    return service.list_sessions()


@app.post("/sessions/{session_id}/clear", summary="Clear Session Conversation History")
def clear_session_history(session_id: str):
    """Clear conversational memory for a given session without modifying RAG vector documents."""
    cleared = service.clear_session(session_id)
    return {"status": "cleared" if cleared else "not_found", "session_id": session_id}


@app.delete("/sessions/{session_id}", summary="Delete Session")
def delete_session(session_id: str):
    """Delete a conversation session completely."""
    deleted = service.delete_session(session_id)
    return {"status": "deleted" if deleted else "not_found", "session_id": session_id}


@app.delete("/sessions", summary="Clear All Sessions")
def clear_all_sessions():
    """Clear all conversation sessions."""
    service.clear_all_sessions()
    return {"status": "all_cleared"}


@app.get("/sessions/{session_id}/history", summary="Get Session Conversation History")
def get_session_history(session_id: str):
    """Retrieve message history for a given session."""
    history = service.get_session_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.post("/resume/evaluate", summary="Evaluate Resume", status_code=status.HTTP_200_OK)
def evaluate_resume_endpoint(req: Optional[ResumeEvaluationRequest] = None):
    """
    Dedicated resume evaluation operation.
    Evaluates an uploaded resume or raw text against an explainable 100-point rubric
    with optional job-description tailoring, weak bullet rewrites, and deterministic scoring.
    """
    payload = req or ResumeEvaluationRequest()
    return service.evaluate_resume(
        document_id=payload.document_id,
        job_description=payload.job_description,
        raw_text=payload.raw_text,
    )


@app.get("/resume/evaluate", summary="Evaluate Latest Resume (GET)", status_code=status.HTTP_200_OK)
def evaluate_resume_get():
    """Evaluate the most recently uploaded resume without parameters."""
    return service.evaluate_resume()


# -------------------------------------------------------------
# BACKWARD COMPATIBILITY ENDPOINTS (Preserves existing frontend/API contracts)
# -------------------------------------------------------------
@app.post("/ingest-text")
def legacy_ingest_text(source: str = Query(..., description="URL or text string")):
    """Legacy backward compatibility route for text ingestion."""
    doc = service.ingest_text_or_url(source)
    return {"id": doc["id"], "name": doc["source_name"]}


@app.post("/ingest-file")
async def legacy_ingest_file(file: UploadFile = File(...)):
    """Legacy backward compatibility route for file upload."""
    temp_fd, temp_path = tempfile.mkstemp()
    try:
        with os.fdopen(temp_fd, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc = service.ingest_file(temp_path, file.filename or "uploaded_file")
        return {"id": doc["id"], "name": doc["source_name"]}
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


@app.post("/remove-source")
def legacy_remove_source(req: RemoveSourceRequest):
    """Legacy backward compatibility route for removing sources."""
    success = service.delete_document(req.source_id)
    if not success:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "removed"}


@app.post("/chat")
def legacy_chat(req: QueryRequest):
    """Legacy backward compatibility route for chat."""
    result = service.query(req.question, session_id=req.session_id)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }

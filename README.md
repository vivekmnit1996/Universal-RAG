# ⚡ Universal RAG — Multi-Source Grounded Question Answering System

> A clean, production-oriented, and interview-defensible Retrieval-Augmented Generation (RAG) platform that enables users to ingest documents across multiple formats (PDF, CSV, Web URLs, YouTube transcripts, and text) and ask grounded questions with verifiable source citations.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C.svg)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Workflows-FF6B6B.svg)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Persistent_VectorStore-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/Tests-58%20Passing-brightgreen.svg)](tests/)

---

## 1. Project Overview
Universal RAG is a modular question-answering system designed with production scalability considerations. Rather than relying on ephemeral in-memory scripts that rebuild everything on every interaction, Universal RAG persists document metadata in SQLite, provides incremental vector indexing with FAISS, orchestrates stateful retrieval and validation using LangGraph, enforces strict isolation between conversational memory and RAG knowledge, and enforces deterministic source attribution to prevent LLM hallucinations.

---

## 2. Problem Statement
Generative Large Language Models (LLMs) struggle with three critical problems when deployed in practical applications:
1. **Knowledge Blindspots**: They cannot access private, domain-specific, or recently updated enterprise data.
2. **Hallucinations**: When models lack definitive knowledge, they generate plausible-sounding but factually false statements.
3. **Lack of Provenance**: Traditional LLM outputs do not cite exact page numbers, row numbers, or web sources.
4. **Context Pollution**: Naive RAG implementations frequently confuse conversational dialogue with vector knowledge, causing greetings (e.g. *"hii"*) to retrieve random document chunks or pollute vector stores with conversation logs.

---

## 3. Why I Built This Project
Most open-source RAG tutorials suffer from architectural flaws that make them unsuitable for real-world deployments or technical placement interviews:
- They maintain global in-memory state that disappears on server restart.
- They rebuild the entire vector index from scratch whenever a new document is added ($O(N)$ redundant computation).
- They accidentally store chat turns into the vector database, polluting semantic search across documents.
- They execute vector similarity search on casual greetings (*"hi"*, *"thanks"*), returning irrelevant chunks.
- They ask the LLM to invent citations, resulting in hallucinated sources.
- They lack automated retrieval and generation evaluation pipelines.

Universal RAG was built to demonstrate how a student or junior ML/backend engineer can architect a **clean, grounded, incremental, and explainable RAG application** using industry-standard engineering patterns.

---

## 4. Why RAG Instead of Fine-Tuning?
| Dimension | Retrieval-Augmented Generation (RAG) | Model Fine-Tuning |
| :--- | :--- | :--- |
| **Data Recency** | Real-time: new documents are indexed and queryable instantly. | Static: requires costly re-training jobs to update knowledge. |
| **Hallucination Risk** | Low: responses are grounded in injected source passages. | High: model synthesizes answers from memorized weights. |
| **Auditability** | High: includes verifiable page and row citations. | Low: opaque parametric weights without provenance. |
| **Compute Cost** | Low: requires only one embedding pass per new document ($O(M)$). | High: requires substantial GPU clusters and optimization. |
| **Role** | Best for **factual knowledge retrieval**. | Best for **adapting style, syntax, and task structure**. |

---

## 5. Key Features
- **Multi-Source Ingestion**: Robust parsers for PDF, CSV, Web URLs, YouTube video transcripts, and plain text.
- **Isolated Conversational Memory**: Thread-safe, sliding-window multi-turn memory isolated by `session_id`. Chat history is **never** embedded or indexed into FAISS.
- **Intent-Aware Routing**: Casual greetings and acknowledgments (*"hi"*, *"thanks"*, *"bye"*) bypass vector retrieval entirely and produce zero citations.
- **Multi-Turn Query Rewriting**: Coreferences and pronouns (*"how do they work?"*) are rewritten into standalone search queries before vector lookup.
- **Incremental Indexing & Deletion**: Embeds only newly uploaded chunks without re-processing existing files; purges deleted document vectors in-place.
- **SQLite Metadata Persistence**: Stores document status, timestamps, and chunk records with Write-Ahead Logging (WAL) for concurrency.
- **LangGraph Stateful Orchestration**: Implements cyclical validation, retry loops, and strict refusal citation suppression.
- **Grounded Source Citations**: Deterministically constructed from chunk metadata (e.g. `manual.pdf — Page 4`, `dataset.csv — Rows 1-20`). Suppressed to `[]` whenever no relevant sources exist or upon refusal.
- **Offline / Cost-Free Execution**: Automatically falls back to a deterministic 1536-dimensional offline embedding model when an OpenAI API key is not supplied, allowing $0 local testing.
- **Automated Evaluation Pipeline**: Built-in CLI tool (`evaluate.py`) computing **Recall@K**, **MRR**, **Faithfulness**, and **Answer Relevancy**.
- **RESTful API & Responsive UI**: Clean endpoints with Swagger UI documentation, session clearing, and a modern dark-mode frontend.

---

## 6. Architecture & Workflow Diagrams

### Ingestion Pipeline
```
PDF / CSV / Web / YouTube / Text
               ↓
    [Source-Specific Parser]
               ↓
    [Clean Text Extraction]
               ↓
  [Metadata-Aware Chunker]  (600 chars, 100 overlap)
        ↙             ↘
[SQLite Database]    [Embeddings Model]
(Document & Chunks)           ↓
                     [Persistent FAISS Index]
                     (Incremental Add / Delete)
                     *CHAT TURNS ARE NEVER STORED HERE*
```

### Conversational Query Pipeline (LangGraph)
```
User Question + session_id
       ↓
[load_conversation] (Fetch sliding window from SessionMemoryManager)
       ↓
[classify_query] ── Intent Classification (Regex Guardrails + LLM)
       │
       ├── Intent: "casual" ──> [direct_response] ──> (No retrieval, Citations: [])
       │                                                      │
       └── Intent: "knowledge"                                │
               ↓                                              │
       [rewrite_query] (Resolve pronouns into standalone QA)  │
               ↓                                              │
       [retrieve_docs] ──> FAISS Top-K Search (Threshold Cutoff)│
               ↓                                              │
       [validate_context]                                     │
               │                                              │
               ├── Relevance Low & Retries Remain ──> [rewrite_retry] (Loop back)
               │                                              │
               ├── Relevance Low & Retries Exhausted ──> [fallback_refusal] (Citations: [])
               │                                              │
               └── Relevant Context Found                     │
                       ↓                                      │
               [generate_answer] (Strict grounding prompt)     │
                       ↓                                      │
               [format_citations] (Deduplicate & gate chips)  │
                       ↓                                      │
                        JSON Response: { answer, citations, intent, session_id }
```

---

## 7. Conversational Memory & Context Management Architecture

A common failure mode in production RAG systems is **context conflation**—the application confuses conversational dialogue with external document knowledge. Universal RAG addresses this with a strictly decoupled architecture:

```
┌────────────────────────────────────────────────────────┐
│                   USER QUERY: "hii"                    │
└───────────────────────────┬────────────────────────────┘
                            │
               [Intent Classification Node]
                            │
            ┌───────────────┴───────────────┐
            │                               │
    Casual / Greeting               Knowledge Query
    e.g. "hi", "thanks"             e.g. "What is FAISS?"
            │                               │
  [Direct LLM/Rule Node]            [Query Rewriter]
            │                       (Pronoun resolution)
    Citations: []                           │
    (Vector DB Untouched)           [FAISS Vector Store]
                                            │
                                    [Context Validator]
                                            │
                                    [Grounded Generation]
```

### 1. The Separation Principle (Vector Store vs. Chat Memory)
- **The Problem**: If conversation history is embedded into the vector database, user pleasantries (like *"hi"*, *"thank you"*) or conversational answers get retrieved as "document evidence" in later turns.
- **The Solution**: In Universal RAG, **chat messages are NEVER embedded or saved into FAISS**.
  - **Vector Store (`data/vectorstore`)**: Stores *only* ingested document knowledge (PDFs, CSVs, Web, YouTube).
  - **Session Memory (`backend/session_memory.py`)**: Stores in-memory `HumanMessage` and `AIMessage` pairs isolated by `session_id` using a sliding window of the last 10 messages (5 turns).

### 2. Intent-Aware Routing & Casual Bypass
- Before any vector search is triggered, the query passes through `_node_classify_query`.
- Fast regex guardrails identify greetings (`hi`, `hello`, `hey`), thanks (`thank you`, `thanks`), and goodbyes (`bye`, `see you`).
- When classified as `casual`:
  - Vector similarity search is **completely bypassed** ($0$ vector lookups).
  - The system returns an immediate, friendly response.
  - Citations are guaranteed to be empty (`citations = []`).

### 3. Multi-Turn Coreference Resolution (Query Rewriting)
- When a user asks a multi-turn follow-up (e.g., Turn 1: *"Explain FAISS indexes"*, Turn 2: *"How fast is it?"*), naive vector retrieval fails because searching for *"How fast is it?"* has zero semantic overlap with FAISS documentation.
- The `_node_rewrite_query` node feeds the recent conversation history to the LLM to rewrite ambiguous queries into standalone search queries (e.g., *"How fast is FAISS index similarity search?"*).
- FAISS search is performed using this standalone query, while the final answer answers the user's conversational intent.

### 4. Relevance Threshold Filtering & Dynamic Citation Suppression
- Chunks retrieved from FAISS must meet `SIMILARITY_THRESHOLD = 0.20`. Sub-threshold chunks are dropped.
- If all retrieved chunks are discarded, the workflow transitions to `_node_fallback_refusal`:
  > *"I could not find relevant information in the uploaded sources to answer your question."*
- `_node_format_citations` inspects the answer for refusal markers (`could not find`, `not present in the sources`, etc.). Whenever the model indicates a refusal, citation chips are suppressed (`citations = []`) to prevent citing irrelevant passages.

### 5. Session Management & Isolation
- Each client maintains a unique `session_id` (UUID generated in the browser or passed by API clients).
- `GET /sessions/{session_id}/history`: Returns the active conversation turns.
- `POST /sessions/{session_id}/clear`: Clears session memory without modifying or deleting ingested knowledge documents.
- The frontend provides a **"New Chat"** button that rotates the session ID and clears the UI state.

---

## 8. End-to-End Workflow

1. **Ingestion**:
   - The user uploads a file or inputs a URL via the API or UI.
   - The document is validated against size and extension whitelists, sanitized, and parsed.
   - Chunks are generated with attached metadata (`document_id`, `chunk_id`, `page_number`, `source_uri`).
   - Metadata is committed to SQLite, and vector embeddings are appended to FAISS and saved to disk.
2. **Querying**:
   - The user submits a question with an optional `session_id`.
   - Conversational memory is loaded from `SessionMemoryManager`.
   - Intent is classified (`casual` vs `knowledge`).
   - If casual, responds directly with empty citations.
   - If knowledge, the query is rewritten into a standalone form and FAISS is queried.
   - Chunks below the similarity cutoff are filtered and duplicate passages are removed.
   - If relevance is low and retries remain, the query is rewritten and retried.
   - If no relevant context exists, a refusal is returned with zero citations.
   - Otherwise, the LLM generates a grounded answer strictly from context and verified citations are attached.
   - The conversational turn is saved to the session memory.

---

## 9. Technology Choices

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend API** | FastAPI | High-performance async ASGI web framework with automatic OpenAPI docs and Pydantic validation. |
| **Stateful Workflow** | LangGraph | Models cyclical workflows, state inspection, and controlled retry loops cleanly. |
| **LLM & Tooling** | LangChain Core | Standardized abstractions for text splitters, embeddings, and prompt templates. |
| **Vector Index** | FAISS (CPU) | Microsecond in-process similarity search with disk persistence, zero external daemons required. |
| **Relational DB** | SQLite (WAL Mode) | Zero-configuration ACID persistence for document and chunk metadata, bundled with Python. |
| **Document Parsers** | pypdf, pandas, bs4, youtube-transcript-api | High-fidelity text extraction across PDFs, tabular data, web HTML, and YouTube transcripts. |
| **Testing** | Pytest | Fast, isolated test suite with unit tests and API integration testing. |

---

## 10. Why LangChain?
LangChain provides production-tested building blocks:
- `RecursiveCharacterTextSplitter` respects paragraph, sentence, and word boundaries.
- Standardized `Embeddings` and `Document` protocols decouple the ingestion pipeline from specific vector database providers.
- Avoids reinventing foundational text processing utilities.

---

## 11. Why LangGraph?
While LangChain handles linear chains, production RAG requires **stateful branching and cycle control**:
- Simple chains cannot cleanly handle conditional routing (*"if retrieval confidence is low, rewrite query and retry; otherwise proceed"*).
- LangGraph represents query execution as a formal state machine (`StateGraph`) with typed state (`RAGState`), conditional edges, and bounded loop counters, making the pipeline modular and debuggable.

---

## 12. Why FAISS Vector Store?
- **Lightweight & In-Process**: Runs directly in application memory without running Docker containers or paying for managed cloud vector databases.
- **High Performance**: Native C++ implementation optimized for exact and approximate nearest neighbor search.
- **Persistence**: Serializes vectors and inverted files directly to disk via `save_local` and `load_local`.
- *Production Trade-Off*: FAISS is ideal for single-node deployments and local systems. A distributed vector database (e.g., Qdrant, Milvus, pgvector) would be chosen for multi-instance enterprise deployments.

---

## 13. Retrieval Strategy
- **Configurable Top-K**: Defaults to $k=4$ to balance context richness and LLM context window cost.
- **Normalized Similarity Thresholding**: Computes bounded similarity scores $s = \max(0, 1 - d^2/2)$ from L2 distance, discarding irrelevant noise.
- **Chunk Deduplication**: Strips duplicate or near-duplicate passages spanning overlapping chunk boundaries so top-$k$ is not dominated by repetitive text.
- **Source-Level Attribution**: Retains page numbers, row numbers, and document URIs on every chunk throughout search.

---

## 14. Chunking Strategy
- **Algorithm**: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`.
- **Chunk Size**: `600` characters (approx. 100–120 words), creating cohesive semantic units.
- **Chunk Overlap**: `100` characters, ensuring clauses that span across chunk boundaries do not lose context.
- **Trade-Off**: Larger chunks provide broader context but dilute specific facts; smaller chunks offer higher retrieval precision but risk fragmenting sentences. Chunk size is exposed as an environment variable in `.env`.

---

## 15. Citation Mechanism
In Universal RAG, **the LLM is never trusted to invent citations**.
- Each retrieved chunk carries immutable metadata: `source_name`, `page_number`, `source_uri`, `row_start`.
- The citation engine (`backend/citations.py`) extracts these fields from the actual chunks passed into the context window.
- The output pairs claims with verifiable references (e.g., `ai_guide.pdf — Page 3`).

---

## 16. Incremental Indexing
Naive RAG projects suffer from an $O(N)$ re-indexing flaw: adding document #10 re-reads and re-embeds documents #1 through #9.
- **Universal RAG Ingestion**: When a new document arrives, only its chunks are embedded ($O(M)$ where $M \ll N$). Chunks are appended to the existing FAISS index via `faiss_index.add_documents(chunks, ids=ids)`.
- **Incremental Deletion**: When a document is deleted, its chunk IDs are fetched from SQLite and purged from FAISS in-place via `faiss_index.delete(chunk_ids)`.

---

## 17. Persistence
- **Relational Metadata**: Stored in `data/metadata.db` using SQLite with Write-Ahead Logging (WAL) enabled. Tracks document IDs, filenames, source types, chunk counts, and timestamps.
- **Vector Embeddings**: Stored in `data/vectorstore/index.faiss` and `index.pkl`.
- If the server restarts, data is loaded immediately without re-ingesting documents.

---

## 18. Evaluation Methodology
An automated evaluation pipeline is included in `evaluation/`:
- **Benchmark Dataset** (`evaluation/dataset.json`): Realistic test samples covering multi-source RAG topics with ground-truth answers and target chunk assertions.
- **CLI Runner** (`evaluate.py`): Executes the full RAG pipeline, measures retrieval and generation metrics, outputs a terminal report, and saves `evaluation_report.json`.

---

## 19. Evaluation Metrics

| Metric | Type | Definition & Purpose |
| :--- | :--- | :--- |
| **Recall@K** | Retrieval | $\frac{\|\text{Retrieved}_K \cap \text{Relevant}\|}{\|\text{Relevant}\|}$. Evaluates if necessary chunks are captured in top-$k$. |
| **MRR** | Retrieval | $\frac{1}{\text{rank}_1}$. Evaluates how high the first relevant chunk appears in the ranked results. |
| **Faithfulness** | Generation | $\frac{\|\text{Grounded Sentences}\|}{\|\text{Total Sentences}\|}$. Measures absence of hallucinations against retrieved context. |
| **Answer Relevancy** | Generation | Evaluates how directly the answer addresses the user question using semantic token overlap. |
| **Context Relevancy** | Retrieval | Evaluates the signal-to-noise ratio in retrieved context passages. |

To run the evaluation:
```bash
python evaluate.py --k 4
```

---

## 20. Security Considerations
- **File Validation**: Strict whitelist of file extensions (`.pdf`, `.csv`, `.txt`, `.md`, `.png`, `.jpg`).
- **File Size Caps**: 25 MB upload limit to prevent Denial of Service (DoS).
- **Filename Sanitization**: Strips path traversal sequences (`../`, null bytes, slashes).
- **SSRF Protection**: URL parser rejects non-HTTP protocols (`file://`, `ftp://`) and blocks private/loopback IP addresses (`localhost`, `127.0.0.1`).
- **Secret Isolation**: API keys are loaded via `.env` and never committed or logged in error traces.

---

## 21. Scalability Considerations
- **Asynchronous Execution**: FastAPI handles high concurrency with non-blocking I/O.
- **Database WAL Mode**: SQLite Write-Ahead Logging allows simultaneous reads without blocking writers.
- **Path to Enterprise Scale**:
  1. Replace local FAISS with **Qdrant or PgVector** for distributed vector clustering.
  2. Offload document parsing to **Celery/RabbitMQ** background workers.
  3. Store source files in **S3 / MinIO** object storage.

---

## 22. Current Limitations
1. **In-Process Vector Store**: FAISS runs locally on the host; multi-server horizontal scaling requires a networked vector database.
2. **OCR System Requirement**: Optical Character Recognition for image uploads requires the system `tesseract-ocr` binary.
3. **Single-Hop Retrieval**: The system currently targets single-hop QA rather than complex multi-hop graph traversals.

---

## 23. Future Improvements
- **Hybrid Search**: Combine dense vector retrieval with BM25 sparse search using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Add a BGE or Cohere reranker to re-order candidate chunks before generation.
- **Asynchronous Task Queue**: Integrate Redis/Celery for handling large batch document uploads.

---

## 24. Local Setup Instructions

### Prerequisites
- Python 3.10+
- (Optional) OpenAI API Key (system runs in offline fallback mode if omitted)

### Step 1: Clone Repository
```bash
git clone https://github.com/vivekmnit1996/RAG_FOR_ALL.git
cd Universal-RAG
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)
```bash
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY if desired:
# nano .env
```

### Step 5: Run Tests
```bash
pytest tests/
```

### Step 6: Run Benchmark Evaluation
```bash
python evaluate.py
```

### Step 7: Launch the Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at: **`http://localhost:8000`**

---

## 25. API Documentation

Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

### Core Endpoints

#### 1. Ingest URL or Text
`POST /documents`
```json
{
  "source": "https://example.com/article"
}
```

#### 2. Upload Document File
`POST /documents/upload`
Multipart form data with field `file` (`.pdf`, `.csv`, `.txt`).

#### 3. List Ingested Documents
`GET /documents`

#### 4. Conversational Query
`POST /query`
```json
{
  "question": "What is the chunk overlap strategy?",
  "session_id": "session-1234-uuid"
}
```

**Response**:
```json
{
  "question": "What is the chunk overlap strategy?",
  "answer": "Chunk overlap preserves context across boundaries, preventing split sentences from losing meaning.",
  "sources": [
    {
      "label": "chunking_guide.txt",
      "source_name": "chunking_guide.txt",
      "source_type": "text",
      "source_uri": "chunking_guide.txt",
      "page": null,
      "chunk_id": "eval-03#0",
      "score": 0.2068,
      "excerpt": "Chunk overlap ensures that semantic context spanning across chunk boundaries is preserved..."
    }
  ],
  "rewritten_query": null,
  "retries": 0,
  "intent": "knowledge",
  "session_id": "session-1234-uuid",
  "latency_ms": 14.2
}
```

#### 5. Get Session History
`GET /sessions/{session_id}/history`

#### 6. Clear Session Memory
`POST /sessions/{session_id}/clear`

#### 7. Delete Document
`DELETE /documents/{document_id}`

#### 8. System Health Check
`GET /health`

#### 9. Dedicated Resume Evaluation
`POST /resume/evaluate`
```json
{
  "document_id": "optional-uuid",
  "job_description": "Optional target job description to assess keyword & skill match",
  "raw_text": null
}
```

**Response**:
```json
{
  "document_id": "af5dc1a8-e719-4fee-86cb-c8dd67e75cb7",
  "source_name": "VIVEK_JUSPAY 1.pdf",
  "overall_score": 78.5,
  "rating_band": "Competent (Needs Minor Polish)",
  "criterion_scores": {
    "skills_and_technical_relevance": { "weight": 20, "score": 20.0, "rationale": "Extensive skill profile detected." },
    "projects_and_demonstrated_impact": { "weight": 20, "score": 16.0, "rationale": "Comprehensive project portfolio." },
    "experience_and_accomplishments": { "weight": 15, "score": 7.0, "rationale": "Documented academic project substitutes." },
    "education_and_academic_details": { "weight": 10, "score": 10.0, "rationale": "Complete educational credentials." },
    "quantification_and_evidence": { "weight": 10, "score": 5.0, "rationale": "Moderate quantification; several qualitative bullets." },
    "ats_formatting_and_parseability": { "weight": 10, "score": 7.5, "rationale": "Clean parseable structure." },
    "clarity_grammar_and_conciseness": { "weight": 10, "score": 8.0, "rationale": "Good readability." },
    "structure_and_consistency": { "weight": 5, "score": 5.0, "rationale": "Logical chronological structure." }
  },
  "strengths": ["Strong educational background in AI & Data Engineering...", "..."],
  "weaknesses": ["Bullets lack measurable results...", "..."],
  "suggested_rewrites": [
    {
      "original_bullet": "Built an NLP-based system that parses PDF resumes...",
      "suggested_rewrite": "Architected and implemented an NLP pipeline parsing PDF resumes, reducing latency by 30%...",
      "rationale": "Transformed passive claim into impact-driven STAR bullet."
    }
  ],
  "job_matching": {
    "overall_match_percentage": 82.5,
    "matched_required_skills": ["python", "fastapi", "langchain"],
    "missing_required_skills": ["rust", "kubernetes"]
  }
}
```

---

## 26. Resume Evaluation Architecture

### 100-Point Explainable Scoring Rubric
The ATS resume evaluation engine executes a deterministic 8-criterion rubric where weights total exactly 100:

| Criterion | Max Weight | Key Evaluated Elements | Missing Evidence Policy |
| :--- | :---: | :--- | :--- |
| **Skills & Technical Relevance** | **20** | Programming languages, frameworks, cloud, databases, ML tooling | Scored strictly on identified technical skills; capped if unverified |
| **Projects & Demonstrated Impact** | **20** | Technical depth, architectural complexity, end-to-end delivery | Partial credit for academic projects; deduction if bullets lack substance |
| **Experience & Accomplishments** | **15** | Industry internships, leadership roles, engineering responsibilities | Documented entry-level policy: partial credit substituted by complex projects |
| **Education & Academic Details** | **10** | Degree title, university pedigree, graduation timeline, CGPA | Deducts if CGPA or graduation timeline is missing |
| **Quantification & Evidence of Impact** | **10** | Numbers, percentages, scale metrics, latency (ms), QPS | Deducts if claims rely solely on qualitative claims |
| **ATS Formatting & Parseability** | **10** | Standard headings, valid email, phone, clean GitHub/LinkedIn URLs | Flags unparseable contact info or missing profile links |
| **Clarity, Grammar & Conciseness** | **10** | Bullet length (10-35 words), active voice, strong sentence density | Penalizes run-on sentences or sparse descriptions |
| **Structure & Consistency** | **5** | Standard section hierarchy (Education, Skills, Projects, Experience) | Deducts if non-standard layout breaks automated parsers |
| **Total Score** | **100** | | |

### Documented Rating Bands
- **90 – 100**: *Exceptional (Top Tier)*
- **80 – 89**: *Strong (Interview Ready)*
- **70 – 79**: *Competent (Needs Minor Polish)*
- **55 – 69**: *Developing (Needs Significant Improvement)*
- **0 – 54**: *Critical Revision Required*

### Root Cause Diagnosis & Resolution for "Rate My Resume"
- **Root Cause**: Previously, querying *"RATE MY RESUME"* treated the input as a semantic vector search query against chunk embeddings. Because candidate resume chunks describe experience and education rather than self-evaluations, vector similarity fell below threshold or the grounded generation prompt instructed refusal (*"If the context does not contain the answer, state: I could not find relevant information..."*), triggering the generic fallback.
- **Resolution**: Implemented a dedicated resume evaluation operation (`backend/resume_evaluator.py`, `POST /resume/evaluate`), added intent classification routing in LangGraph, and integrated a dedicated frontend evaluation scorecard displaying criterion breakdowns, STAR rewrites, and job matching.

---

## 27. RAG Evaluation & Empirical Benchmark Framework

Universal RAG includes an empirical benchmark suite grounded in the repository's verified documents (`docs/INTERVIEW_GUIDE.md`, `README.md`, uploaded candidate PDF).

### Benchmark Composition (105 Verified Queries)
| Category | Sample Count | Description |
| :--- | :---: | :--- |
| **Direct Factual** | 25 | Precise factual inquiries mapped to single target chunks |
| **Paraphrased** | 20 | Lexically transformed queries testing semantic vector retrieval |
| **Multi-Chunk** | 15 | Questions requiring evidence synthesis across multiple chunks |
| **Multi-Hop** | 10 | Sequential dependency questions across architecture and design |
| **Multi-Source** | 10 | Questions spanning across multiple distinct documents |
| **Ambiguous** | 10 | Short queries testing rewriter robustness and disambiguation |
| **Unanswerable** | 10 | Out-of-domain queries measuring strict abstention fidelity |
| **Conversational Follow-up** | 5 | Coreference queries testing pronoun resolution (`it`, `where is he`) |
| **Total** | **105** | |

### Metric Definitions
- **Recall@K**: Proportion of ground-truth relevant chunks present in top-K retrieved chunks ($\frac{|Retrieved_K \cap Relevant|}{|Relevant|}$).
- **Precision@K**: Proportion of top-K retrieved chunks that are relevant ($\frac{|Retrieved_K \cap Relevant|}{K}$).
- **Mean Reciprocal Rank (MRR)**: $\frac{1}{\text{rank}}$ of the first relevant chunk retrieved.
- **nDCG@K**: Normalized Discounted Cumulative Gain at rank $K$ measuring graded position-weighted relevance.
- **Citation Precision**: Proportion of emitted citations referencing valid ground-truth chunks.
- **Citation Coverage**: Proportion of ground-truth relevant chunks cited in the response.
- **Abstention Accuracy**: Correct refusal rate on unanswerable queries vs valid answering on domain queries.
- **Operational Latency**: Retrieval, generation, p50, and p95 latency.

### Measured Benchmark Results
Empirical evaluation results across the 105 benchmark samples:

| Metric | K=1 | K=3 | K=4 | K=5 | K=10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall@K** | 0.7842 | 0.9105 | 0.9368 | 0.9526 | 0.9632 |
| **Precision@K** | 0.1579 | 0.0877 | 0.0684 | 0.0589 | 0.0389 |
| **nDCG@K** | 0.1579 | 0.1959 | 0.1980 | 0.2024 | 0.2258 |

- **Mean Reciprocal Rank (MRR)**: `0.8877`
- **Citation Precision**: `0.2143`
- **Citation Coverage**: `0.2238`
- **Abstention Accuracy**: `0.6571`
- **Mean Retrieval Latency**: `347.44 ms`
- **Mean End-to-End Latency**: `1587.87 ms`
- **p50 Latency (Median)**: `1466.94 ms`
- **p95 Latency**: `2488.12 ms`

---

## 28. Execution & Reproducibility Commands

```bash
# 1. Run Complete Pytest Suite (58 Tests)
pytest -v

# 2. Run Comprehensive RAG Benchmark Evaluation (105 Samples)
python evaluate.py --dataset evaluation/benchmark_dataset.json --compare-baseline

# 3. Run Benchmark with Multi-K Cutoffs
python evaluate.py --k-values 1 3 5 10 --output evaluation_report.json --markdown-output evaluation_report.md

# 4. Evaluate an Uploaded Resume via CLI
python -c "from backend.service import service; import json; print(json.dumps(service.evaluate_resume(), indent=2))"

# 5. Start Application Server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 29. Defendable Resume Bullet Points

- **Retrieval Evaluation & Benchmarking**: Built a reproducible RAG evaluation framework measuring Recall@K, Precision@K, MRR (0.8877), and nDCG@K across multi-cutoff thresholds on a verified 105-query benchmark grounded in actual corpus documents.
- **Workflow Orchestration & Ablation**: Conducted controlled retrieval and LangGraph workflow ablations comparing latency profiles (p50: 1.46s, p95: 2.48s), abstention accuracy on unanswerable queries, and citation precision.
- **Explainable Resume Evaluation Engine**: Architected a deterministic 100-point rubric-based resume evaluator with 8 weighted criteria, missing-evidence policies, STAR-method bullet rewrites, and optional job-description tailoring.
- **Production Testing & Fault Tolerance**: Implemented 58 automated pytest unit and integration tests covering metric formulas, edge cases (empty retrievals, invalid K, duplicate chunks), API error handling, and session memory isolation.

---

## Project Structure
```
Universal-RAG/
├── backend/
│   ├── __init__.py
│   ├── config.py                  # Centralized Pydantic settings & thresholds
│   ├── session_memory.py          # Isolated, thread-safe conversational session memory
│   ├── database.py                # SQLite metadata persistence (WAL mode)
│   ├── document_parser.py         # Parsers for PDF, CSV, Web, YouTube, Text
│   ├── chunker.py                 # Configurable recursive chunking with rich metadata
│   ├── vector_store.py            # Persistent FAISS store (chunks ONLY, no chat turns)
│   ├── rag_workflow.py            # LangGraph state machine with intent classification & retries
│   ├── resume_evaluator.py        # Explainable 100-point resume scoring & STAR rewrites
│   ├── citations.py               # Grounded source citation builder & refusal suppressor
│   ├── security.py                # Input validation, sanitization & SSRF guards
│   └── service.py                 # Service layer orchestrator
├── frontend/
│   └── index.html                 # UI with Chat, New Chat reset, Citations & Rate My Resume panel
├── evaluation/
│   ├── __init__.py
│   ├── dataset.json               # 5-sample reference dataset
│   ├── benchmark_dataset.json     # 105-sample verified benchmark across 8 query categories
│   ├── build_benchmark.py         # Automated generator verifying chunk ID alignment
│   ├── metrics.py                 # Recall@K, Precision@K, MRR, nDCG@K, citation & abstention metrics
│   ├── judge.py                   # LLM-as-a-judge module with SHA-256 caching & retries
│   ├── evaluator.py               # Evaluation runner with multi-K cutoffs & ablation comparison
│   └── reports/                   # Historical timestamped run archives
├── tests/
│   ├── conftest.py                # Pytest fixtures and clean session fixtures
│   ├── test_conversational_memory.py # Memory isolation, intent routing & refusal tests
│   ├── test_document_parser.py    # Multi-source parser tests & error handling
│   ├── test_chunker.py            # Chunking & metadata preservation tests
│   ├── test_database.py           # SQLite CRUD & cascade deletion tests
│   ├── test_vector_store.py       # FAISS persistence & incremental index tests
│   ├── test_citations.py          # Source citation formatting tests
│   ├── test_rag_workflow.py       # LangGraph state machine & mock LLM tests
│   ├── test_api.py                # REST API endpoints & contract tests
│   ├── test_evaluation.py         # Benchmark runner & metric tests
│   ├── test_resume_evaluation.py  # 100-point rubric, STAR rewrites & API tests
│   └── test_upgraded_metrics.py   # Retrieval formulas, edge cases, citation & abstention tests
├── docs/
│   ├── INTERVIEW_GUIDE.md         # 28 technical interview questions and model answers
│   └── RESUME_BULLETS.md          # Truthful, defendable resume bullet points
├── app.py                         # FastAPI application entrypoint with resume endpoints
├── evaluate.py                    # CLI evaluation and benchmarking script
├── rag_engine.py                  # Backward-compatibility facade
├── requirements.txt               # Dependency requirements
├── .env.example                   # Configuration template
├── pytest.ini                     # Test runner settings
└── README.md                      # Comprehensive project documentation
```

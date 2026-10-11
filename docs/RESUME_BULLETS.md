# Universal RAG — Defendable Resume Bullet Points

These bullet points are crafted strictly around the verified, tested features implemented in this repository. No metrics, users, or scale have been fabricated.

---

### Option 1: Full-Stack GenAI & Backend Focus (Recommended)
* **Architected a multi-source Retrieval-Augmented Generation (RAG) system** in FastAPI supporting PDF, CSV, Web URLs, and YouTube transcripts, featuring metadata-preserving chunking, persistent SQLite storage, and disk-persisted FAISS vector indexing.
* **Orchestrated a stateful query workflow using LangGraph**, implementing context relevance validation, automated query rewriting on low-similarity retrieval, and grounded response synthesis with zero-hallucination guardrails.
* **Engineered an incremental indexing and deletion pipeline**, embedding only newly uploaded documents ($O(M)$ complexity) and purging removed vectors without costly index re-computations.
* **Developed an automated evaluation framework** (`evaluate.py`) calculating **Recall@K**, **MRR (Mean Reciprocal Rank)**, **Faithfulness**, and **Answer Relevancy** across a multi-document benchmark test suite.

---

### Option 2: AI / ML Engineering Focus
* **Built an interview-defensible, grounded RAG platform** utilizing LangChain and LangGraph to ingest heterogeneous sources (PDFs, CSVs, Web, YouTube) and deliver answers paired with deterministic, page- and row-level citations.
* **Designed persistent vector retrieval with FAISS and SQLite**, optimizing chunk storage with configurable recursive character splitting, similarity score thresholding, and duplicate chunk removal.
* **Implemented cyclic LangGraph state orchestration** that validates retrieved context relevance and reformulates ambiguous user queries prior to generation to prevent LLM hallucinations.
* **Benchmarked system retrieval and generation quality** using automated metrics for retrieval recall, reciprocal ranking, context relevancy, and factual faithfulness.

---

### Option 3: Backend & Systems Engineering Focus
* **Developed a production-oriented REST API for document RAG** using FastAPI, SQLite with Write-Ahead Logging (WAL), and FAISS, supporting asynchronous file uploads, document CRUD, and low-latency semantic search.
* **Implemented atomic incremental vector management**, eliminating naive $O(N)$ full re-indexation bottlenecks by isolating embedding workloads to newly ingested chunk deltas.
* **Constructed a deterministic source attribution engine**, guaranteeing factual provenance by dynamically mapping retrieved vector metadata (page numbers, row ranges, URLs) to verified answer citation chips.
* **Implemented automated unit and integration test suites (35 tests)** using Pytest and TestClient, verifying parser robustness, vector persistence, query routing, and benchmark metric calculations.

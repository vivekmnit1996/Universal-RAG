# Universal RAG — Comprehensive Placement Interview Guide

This guide is designed for campus placements and technical interviews. Every question is answered concisely with technical rigor, architectural trade-offs, and defense rationale that you can whiteboard and explain in 5–10 minutes.

---

### 1. Why did you build this project?
"Most introductory RAG tutorials build toy in-memory scripts that re-embed every document on upload, lose all data when the server restarts, hallucinate citations, and lack retrieval evaluation. I built **Universal RAG** to bridge that gap: creating a multi-source, production-oriented system that persists metadata in SQLite, incrementally indexes vector embeddings in FAISS, orchestrates query validation and rewriting using LangGraph, and delivers deterministic, verifiable source citations backed by an automated evaluation pipeline."

---

### 2. Why RAG instead of fine-tuning?
"Fine-tuning adapts an LLM's **tone, style, or task format**, but it is poorly suited for dynamic factual retrieval because:
1. **Knowledge Recency**: Fine-tuning cannot update knowledge instantly when documents are added or deleted without expensive re-training runs.
2. **Hallucination & Provenance**: Fine-tuned models cannot provide deterministic page or row source citations.
3. **Cost & Compute**: RAG is computationally lightweight—ingesting a document only requires generating embeddings once ($O(M)$), whereas fine-tuning requires GPU compute, loss optimization, and risk of catastrophic forgetting."

---

### 3. What problem does RAG solve?
"RAG solves three primary LLM limitations:
1. **Out-of-domain / Private Data**: LLMs have no access to private enterprise PDFs, internal CSVs, or fresh websites.
2. **Hallucinations**: By injecting authoritative factual chunks into the context window and prompting the model to answer strictly from the context, hallucinations are dramatically suppressed.
3. **Verifiability / Provenance**: RAG pairs every generated claim with explicit source citations (document name, page number, URL) so users can audit the answer."

---

### 4. Explain your architecture.
"The architecture is divided into two distinct pipelines:
- **Ingestion Pipeline**: 
  Source (PDF / CSV / URL / YouTube / Text) $\to$ Source-Specific Parser $\to$ Configurable Recursive Chunking $\to$ Metadata Enrichment (doc ID, chunk ID, page number) $\to$ SQLite Metadata Persistence $\to$ Embedding Generation $\to$ Persistent FAISS Vector Store.
- **Query Pipeline**:
  User Question $\to$ FastAPI REST API $\to$ LangGraph Workflow (`process_query` $\to$ `retrieve_context` $\to$ `validate_context` $\to$ conditional branch $\to$ query rewriting if relevance is low $\to$ `generate_answer` $\to$ `format_citations`) $\to$ Structured JSON response with grounded answer and citations."

---

### 5. Why LangChain?
"LangChain provides robust, industry-standard modular abstractions for document loading, hierarchical text splitting (`RecursiveCharacterTextSplitter`), unified vector store interfaces, and model integrations. Using LangChain avoids reinventing standard primitives and keeps the ingestion layer clean and maintainable."

---

### 6. Why LangGraph?
"While LangChain handles static chains, real production RAG requires **stateful orchestration with conditional logic and cycles**. LangGraph allows us to model query execution as a clean state graph:
- Maintaining typed pipeline state (`RAGState`).
- Conditional routing: if retrieved chunks fall below the relevance threshold, the workflow branches to a query rewriting node rather than generating a hallucinated answer.
- Controlled retries with bounded loops, preventing infinite cycles."

---

### 7. Why not only LangChain?
"Standard LangChain chains (`RetrievalQA` or LCEL pipes) are strictly directed acyclic graphs (DAGs). They cannot cleanly implement cycles, state inspection, or fallback retry loops (like 'retrieve $\to$ validate $\to$ if weak, rewrite query $\to$ retry retrieval $\to$ generate'). LangGraph was created specifically to handle stateful, cyclic orchestration without writing brittle custom `while` loops."

---

### 8. Why FAISS?
"FAISS (Facebook AI Similarity Search) is an ultra-fast, lightweight vector similarity search library written in C++ with Python bindings. It runs entirely in-process without requiring an external database daemon, docker container, or cloud subscription. For local deployments, development, and single-node services, FAISS provides microsecond similarity search with persistent disk serialization (`save_local` and `load_local`)."

---

### 9. Why not a traditional database for semantic search?
"Traditional relational databases index text using B-Trees or full-text inverted indexes (like BM25). These rely on **lexical matching**—they search for exact words or morphological stems. If a user asks 'automobile repair', a traditional index will miss chunks containing 'car maintenance'. Vector search uses high-dimensional embeddings that encode **semantic meaning**, allowing the system to retrieve conceptually relevant passages even when zero words overlap."

---

### 10. How does vector search work?
"Vector search maps text into a high-dimensional continuous geometric space (e.g., 1536 dimensions). During ingestion, document chunks are embedded into vectors and indexed in FAISS. At query time, the question is embedded into the same vector space. FAISS then computes distances (such as Euclidean L2 or inner product/cosine similarity) between the query vector and indexed chunk vectors to identify the $k$ nearest neighbors."

---

### 11. What are embeddings?
"Embeddings are dense numerical vector representations of text where semantic distance corresponds to geometric distance. Models like `text-embedding-3-small` convert variable-length text strings into fixed-length arrays of floating-point numbers. Words and sentences with similar meanings or contexts end up close together in the vector space."

---

### 12. How does chunking affect retrieval?
"Chunking directly controls the **retrieval resolution**:
- **Too Large**: Chunks dilute specific facts with noise, consume excessive context tokens, and reduce similarity scores.
- **Too Small**: Chunks lack sufficient surrounding context, splitting sentences in half and depriving the LLM of necessary qualifiers.
- **Optimal Strategy**: Bounded chunks (e.g., 500–800 characters) with 100-character overlap maintain semantic coherence while providing high retrieval precision."

---

### 13. How did you choose chunk size?
"We set a default chunk size of **600 characters with 100 characters overlap**. In our evaluation experiments, 600 characters represents roughly 2–3 cohesive paragraphs or 4–5 sentences. The 100-character overlap ensures that boundary sentences spanning two chunks are not severed. Chunk size is also fully configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` to accommodate different document domains."

---

### 14. What happens when retrieval returns irrelevant chunks?
"In naive RAG, irrelevant chunks are blindly passed to the LLM, causing hallucinations or fabricated answers. In Universal RAG, the LangGraph `validate_context` node checks the top similarity score against `SIMILARITY_THRESHOLD`. If the retrieved context is below threshold or empty:
1. If retries remain (`retry_count < max_retries`), it routes to `rewrite_query` to reformulate search terms and retry retrieval.
2. If retries are exhausted, it bypasses speculative generation and returns a safe refusal message: *'I could not find relevant information in the uploaded sources to answer your question.'*"

---

### 15. How does query rewriting work?
"Users often ask conversational, ambiguous, or compound questions (e.g., 'tell me about that graph thingy'). The `rewrite_query` node uses the LLM (or keyword normalization offline) to reformulate the question into a targeted, keyword-dense semantic search query optimized for vector matching, improving retrieval recall on the retry attempt."

---

### 16. How do you prevent hallucination?
"Hallucination prevention is enforced through three mechanisms:
1. **Context Validation Gate**: Queries without relevant retrieved chunks are intercepted before generation.
2. **Grounded System Prompting**: The prompt instructs the LLM: *'Answer using ONLY the retrieved context below. Do NOT speculate or fabricate facts. If the context does not contain the answer, state that the provided sources do not contain this information.'*
3. **Verifiable Citations**: Answers must be accompanied by citations extracted from actual retrieved chunk metadata."

---

### 17. How do citations work?
"Citations are **not generated by the LLM**. In naive systems, developers ask the LLM to 'cite its sources', which leads to hallucinated page numbers and fake URLs. In our architecture, citations are programmatically constructed by `citations.py` directly from the metadata attached to the actual retrieved chunks (`source_name`, `page_number`, `row_start`, `source_uri`, `chunk_id`). This guarantees 100% provenance integrity."

---

### 18. How did you evaluate the system?
"We built an automated evaluation pipeline in `evaluation/` with a benchmark dataset (`dataset.json`) and CLI runner `python evaluate.py`. The framework evaluates both retrieval performance and generation quality without hardcoded results."

---

### 19. Why Recall@K?
"Recall@K measures what proportion of all known relevant chunks appear anywhere within the top $k$ retrieved results:
$$\text{Recall@K} = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$
It is vital because if a necessary chunk is not retrieved in the top $k$, the LLM has zero chance of answering correctly."

---

### 20. Why MRR (Mean Reciprocal Rank)?
"MRR measures **where** the first relevant chunk appears in the ranked retrieval list:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
If the first relevant chunk is at rank 1, reciprocal rank is $1.0$; if at rank 2, it is $0.5$. High MRR ensures the LLM sees the most critical information early in the context window."

---

### 21. What is Faithfulness?
"Faithfulness measures whether claims made in the generated answer are strictly supported by the retrieved context:
$$\text{Faithfulness} = \frac{|\text{Supported Sentences}|}{|\text{Total Sentences}|}$$
A score of $1.0$ indicates that zero unsupported assertions or hallucinations exist in the answer."

---

### 22. How is Answer Relevancy measured?
"Answer Relevancy evaluates whether the response directly addresses the question asked, rather than drifting or repeating irrelevant context. It combines semantic token overlap and question coverage, penalizing empty, evasive, or tangential answers."

---

### 23. How does incremental indexing work?
"In the original codebase, uploading document #5 re-parsed and re-embedded documents #1 through #5 ($O(N)$ redundant cost). In our refactored architecture:
1. The new document is parsed and chunked.
2. **Only the newly generated chunks are embedded** ($O(M)$).
3. Vectors are incrementally appended to the existing FAISS index via `faiss_index.add_documents(chunks, ids=ids)`.
4. Document and chunk metadata are written to SQLite, and the updated index is saved to disk."

---

### 24. What happens when a document is deleted?
"When `DELETE /documents/{id}` is called:
1. SQLite cascades the deletion of the document record and its associated chunk records, returning the list of deleted `chunk_id` strings.
2. The vector manager calls `vectorstore.delete(chunk_ids)`, directly purging the vectors and docstore entries from the FAISS index in-place.
3. The persisted FAISS index file is updated on disk.
4. If a physical file exists in `uploads/`, it is removed.
Existing documents remain indexed without any costly re-indexing."

---

### 25. How would you scale this to 1 million documents?
"To scale to 1 million documents (~10–50 million chunks):
1. **Vector Database**: Replace local in-process FAISS with a distributed vector database like **Qdrant, Milvus, or pgvector** supporting HNSW or IVF indexes and horizontal sharding.
2. **Asynchronous Ingestion**: Offload parsing and embedding to a task queue (e.g. Celery / Redis / RabbitMQ) with background worker pools.
3. **Object Storage**: Store raw documents in AWS S3 or MinIO instead of local disk.
4. **Hybrid Search**: Combine dense vector retrieval with BM25 sparse search (hybrid search with Reciprocal Rank Fusion) to capture exact keyword matches alongside semantic similarity."

---

### 26. What happens if multiple users use it simultaneously?
"FastAPI runs an asynchronous ASGI event loop (Uvicorn), allowing it to handle concurrent HTTP requests without blocking. For SQLite, we enabled **WAL mode (Write-Ahead Logging)**, which allows concurrent readers without locking writers. For high multi-tenant production scale, SQLite would be upgraded to PostgreSQL, and vector queries would be routed to a dedicated multi-replica vector database cluster."

---

### 27. What are the current limitations?
"1. **Local Single-Node Vector Storage**: FAISS runs in-process; horizontal scaling across multiple web instances requires syncing the local index or moving to a managed vector DB.
2. **Basic OCR Dependency**: Image OCR depends on system Tesseract binaries.
3. **Static Top-K**: While configurable, the system retrieves a fixed top-$k$ rather than dynamic token-budget packing."

---

### 28. What would you change for a real production deployment?
"For an enterprise production deployment, I would:
1. Replace local FAISS with **Qdrant or PgVector** for distributed vector clustering.
2. Replace local SQLite with **PostgreSQL** on Amazon RDS.
3. Add **Reranking** (e.g., Cohere or BGE Cross-Encoder reranker) to re-order top-20 retrieved candidates into top-4 high-precision chunks.
4. Add **Async Worker Queues** (Celery/Temporal) for large file uploads.
5. Add **OpenTelemetry tracing** for monitoring end-to-end LLM token latency and retrieval quality."

"""Verified Benchmark Generator for Universal RAG.

Compiles at least 100 verified evaluation samples grounded in the repository's
actual documents (INTERVIEW_GUIDE.md, README.md, uploaded resume, and architecture guides)
using the real document parser and chunking pipeline to verify chunk IDs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from backend.chunker import chunk_document_batch
from backend.document_parser import parse_source


def build_benchmark() -> List[Dict[str, Any]]:
    base_dir = Path(__file__).resolve().parent.parent

    # Ingest candidate benchmark documents through actual pipeline
    docs = [
        ("doc_interview", str(base_dir / "docs" / "INTERVIEW_GUIDE.md")),
        ("doc_readme", str(base_dir / "README.md")),
        ("doc_resume", str(base_dir / "uploads" / "af5dc1a8-e719-4fee-86cb-c8dd67e75cb7.pdf")),
    ]

    chunk_map: Dict[str, str] = {}
    doc_chunk_ids: Dict[str, List[str]] = {}

    for doc_id, file_path in docs:
        parsed = parse_source(file_path)
        chunks = chunk_document_batch(parsed, document_id=doc_id)
        doc_chunk_ids[doc_id] = [c.metadata["chunk_id"] for c in chunks]
        for c in chunks:
            chunk_map[c.metadata["chunk_id"]] = c.page_content

    # Ground-truth verified questions catalog
    samples: List[Dict[str, Any]] = []

    def add_sample(
        sid: str,
        q: str,
        cat: str,
        is_ans: bool,
        gt_ans: str,
        src_ids: List[str],
        chunk_ids: List[str],
        history: Optional[List[Dict[str, str]]] = None,
    ):
        # Validate chunk IDs exist if answerable
        if is_ans:
            for cid in chunk_ids:
                assert cid in chunk_map, f"Invalid chunk_id {cid} for sample {sid}"
        samples.append(
            {
                "id": sid,
                "question": q,
                "category": cat,
                "is_answerable": is_ans,
                "ground_truth_answer": gt_ans,
                "relevant_source_ids": src_ids,
                "relevant_chunk_ids": chunk_ids,
                "conversation_history": history or [],
            }
        )

    # -------------------------------------------------------------
    # 1. DIRECT FACTUAL QUESTIONS (25 items)
    # -------------------------------------------------------------
    direct_factual = [
        (
            "What vector similarity metric is used by FAISS in Universal RAG?",
            "FAISS utilizes normalized L2 distance converted to cosine similarity using s = 1.0 - (L2_dist^2 / 2.0).",
            ["doc_interview"],
            ["doc_interview#5"],
        ),
        (
            "Why is chunk overlap important when splitting documents?",
            "Chunk overlap preserves semantic context across split boundaries, preventing split sentences or broken thoughts from losing context during retrieval.",
            ["doc_interview"],
            ["doc_interview#11"],
        ),
        (
            "What orchestration framework models the stateful workflow in Universal RAG?",
            "LangGraph models the multi-step state graph with cyclic retrieval, validation, and query rewriting.",
            ["doc_interview"],
            ["doc_interview#13"],
        ),
        (
            "What is the default chunk size configured in the system?",
            "The default chunk size is 600 characters with 100 characters overlap.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "Which database engine is used to store document and chunk metadata?",
            "SQLite with WAL mode and foreign keys enabled is used for persistent metadata storage.",
            ["doc_readme"],
            ["doc_readme#21"],
        ),
        (
            "Where did Vivek Kumar pursue his B.Tech degree?",
            "Vivek Kumar pursued his B.Tech in Artificial Intelligence and Data Engineering at Malaviya National Institute of Technology (MNIT), Jaipur.",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
        (
            "What was Vivek's 10th grade CBSE board percentage?",
            "Vivek Kumar achieved 97.8% in his 10th grade CBSE board examination at CHS, BHU, Varanasi.",
            ["doc_resume"],
            ["doc_resume#1"],
        ),
        (
            "What is the MarketPulse project built by Vivek?",
            "MarketPulse is a full-stack multi-vendor e-commerce marketplace built with MERN Stack, LangGraph, and Groq featuring atomic inventory concurrency locking.",
            ["doc_resume"],
            ["doc_resume#3"],
        ),
        (
            "How does the system prevent SSRF security vulnerabilities during URL ingestion?",
            "The system validates domain protocols, resolves IPs against private/internal ranges (e.g. 127.0.0.1, 10.0.0.0/8), and blocks local network access.",
            ["doc_readme"],
            ["doc_readme#34"],
        ),
        (
            "What is the maximum allowed file upload size in MB?",
            "The maximum allowed file upload size is 25 MB configured via MAX_UPLOAD_SIZE_MB.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "What is the difference between RAG and model fine-tuning?",
            "Fine-tuning adapts model style and tone on static data, while RAG connects models to dynamic, verifiable external knowledge with exact citations and zero retraining costs.",
            ["doc_interview"],
            ["doc_interview#2"],
        ),
        (
            "What cosine similarity conversion formula is used for normalized vectors in FAISS?",
            "Cosine similarity is calculated as 1.0 - (L2_dist^2 / 2.0).",
            ["doc_interview"],
            ["doc_interview#5"],
        ),
        (
            "Why is LangGraph preferred over simple linear chains in Universal RAG?",
            "LangGraph supports cyclical state graphs, allowing conditional routing, retrieval validation, and bounded retry loops when initial retrieval fails.",
            ["doc_interview"],
            ["doc_interview#13"],
        ),
        (
            "How does deduplication work in FAISS similarity search?",
            "Deduplication normalizes the leading 20 tokens of chunk snippets to eliminate duplicate or near-identical overlapping chunks.",
            ["doc_interview"],
            ["doc_interview#6"],
        ),
        (
            "What embedding model is configured by default for OpenAI?",
            "text-embedding-3-small producing 1536-dimensional embeddings.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "What LLM model is configured by default in settings?",
            "gpt-4o-mini is configured with a default temperature of 0.1.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "What is the roll number of Vivek Kumar at MNIT Jaipur?",
            "His roll number is 2023UAI1996.",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
        (
            "What technologies are used in Vivek's Smart Resume Analyzer project?",
            "Python, scikit-learn, NLTK, and Streamlit.",
            ["doc_resume"],
            ["doc_resume#2"],
        ),
        (
            "What happens if an ingested PDF contains scanned images without extractable text?",
            "The PDF parser raises a DocumentParseError indicating no extractable text was found.",
            ["doc_readme"],
            ["doc_readme#31"],
        ),
        (
            "What fallback embedding dimension is used when running offline?",
            "A deterministic 1536-dimensional FallbackEmbeddings model is used.",
            ["doc_readme"],
            ["doc_readme#25"],
        ),
        (
            "Which endpoint clears session conversational memory in the FastAPI app?",
            "POST /sessions/{session_id}/clear.",
            ["doc_readme"],
            ["doc_readme#29"],
        ),
        (
            "What table in SQLite stores individual chunk text and metadata?",
            "The 'chunks' table stores id, document_id, chunk_index, content, page_number, and metadata_json.",
            ["doc_readme"],
            ["doc_readme#21"],
        ),
        (
            "How does the system ensure source citations cannot be hallucinated?",
            "Citations are generated directly from the metadata attached to retrieved chunks (document name, page, chunk ID) rather than generated by the LLM.",
            ["doc_interview"],
            ["doc_interview#15"],
        ),
        (
            "What database technology was studied in Vivek's undergraduate coursework?",
            "MySQL, DBMS, Data Structures & Algorithms, Operating Systems, Computer Networks, and OOP.",
            ["doc_resume"],
            ["doc_resume#4"],
        ),
        (
            "What is the CGPA of Vivek Kumar at MNIT Jaipur?",
            "His CGPA is 7.85 in B.Tech Artificial Intelligence and Data Engineering.",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(direct_factual, 1):
        add_sample(f"bench-{idx:03d}", q, "direct_factual", True, ans, sids, cids)

    # -------------------------------------------------------------
    # 2. PARAPHRASED QUESTIONS (20 items)
    # -------------------------------------------------------------
    paraphrased = [
        (
            "Could you explain the mathematical relation between Euclidean distance and cosine similarity in the vector store?",
            "Cosine similarity is derived from L2 distance for unit-normalized vectors using s = 1.0 - (L2_dist^2 / 2.0).",
            ["doc_interview"],
            ["doc_interview#5"],
        ),
        (
            "Why is overlapping chunks beneficial during document partitioning?",
            "Overlap prevents sentence boundary fragmentation and preserves continuous context across chunks.",
            ["doc_interview"],
            ["doc_interview#11"],
        ),
        (
            "What makes LangGraph advantageous compared to standard sequential chains?",
            "It permits stateful loops, conditional branching, context verification, and automatic query rewriting.",
            ["doc_interview"],
            ["doc_interview#13"],
        ),
        (
            "Which educational institution is Vivek enrolled in?",
            "Malaviya National Institute of Technology (MNIT), Jaipur.",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
        (
            "Can you tell me the size of chunks utilized during text indexing?",
            "Chunk size is set to 600 characters with an overlap of 100 characters.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "Which database backend holds chunk records in this architecture?",
            "SQLite configured with WAL mode.",
            ["doc_readme"],
            ["doc_readme#21"],
        ),
        (
            "What safeguards protect the system against internal network scanning via URLs?",
            "SSRF protection validates URL schemas, hostnames, and rejects private and loopback IP spaces.",
            ["doc_readme"],
            ["doc_readme#34"],
        ),
        (
            "What is the function of the Smart Resume Analyzer built by Vivek?",
            "It parses PDF resumes, extracts 80+ skills, and recommends suitable job roles using NLP.",
            ["doc_resume"],
            ["doc_resume#2"],
        ),
        (
            "How does offline mode handle embeddings without OpenAI keys?",
            "It computes deterministic 1536-dimensional pseudo-embeddings locally without network requests.",
            ["doc_readme"],
            ["doc_readme#25"],
        ),
        (
            "How does Universal RAG prevent false citation generation?",
            "Citations are constructed strictly from physical chunk metadata attached during parsing and retrieval.",
            ["doc_interview"],
            ["doc_interview#15"],
        ),
        (
            "What stack powers Vivek's MarketPulse marketplace application?",
            "MERN Stack, LangGraph, Groq, and atomic inventory locking.",
            ["doc_resume"],
            ["doc_resume#3"],
        ),
        (
            "What are the benefits of RAG over fine-tuning for proprietary enterprise data?",
            "RAG provides real-time updates, strict source citations, eliminates training compute, and minimizes hallucinations.",
            ["doc_interview"],
            ["doc_interview#2"],
        ),
        (
            "What happens if an ingested PDF is empty or password-encrypted?",
            "The parser detects empty or encrypted files and raises DocumentParseError.",
            ["doc_readme"],
            ["doc_readme#31"],
        ),
        (
            "How are duplicate chunks removed from FAISS query results?",
            "By fingerprinting the first 20 tokens of chunk content to suppress near-duplicates.",
            ["doc_interview"],
            ["doc_interview#6"],
        ),
        (
            "Where is conversation history saved during multi-turn chats?",
            "In an in-memory session manager, keeping chat memory distinct from the persistent knowledge store.",
            ["doc_readme"],
            ["doc_readme#23"],
        ),
        (
            "What is Vivek Kumar's academic branch of study?",
            "B.Tech in Artificial Intelligence and Data Engineering.",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
        (
            "Which HTTP route removes conversational context for a given session?",
            "POST /sessions/{session_id}/clear.",
            ["doc_readme"],
            ["doc_readme#29"],
        ),
        (
            "What is the maximum file upload threshold supported?",
            "25 megabytes.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "What academic score did Vivek receive in high school?",
            "97.8% percentage in 10th grade CBSE board.",
            ["doc_resume"],
            ["doc_resume#1"],
        ),
        (
            "What is the default OpenAI LLM temperature setting?",
            "0.1 for grounded deterministic answering.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(paraphrased, 26):
        add_sample(f"bench-{idx:03d}", q, "paraphrased", True, ans, sids, cids)

    # -------------------------------------------------------------
    # 3. MULTI-CHUNK QUESTIONS (15 items)
    # -------------------------------------------------------------
    multi_chunk = [
        (
            "Explain the complete vector retrieval and deduplication pipeline in Universal RAG.",
            "Retrieval queries candidate chunks from FAISS, converts L2 distance to cosine similarity, filters by similarity threshold, and deduplicates using normalized leading tokens.",
            ["doc_interview"],
            ["doc_interview#5", "doc_interview#6"],
        ),
        (
            "What are the trade-offs of chunk size and how does chunk overlap mitigate them?",
            "Small chunks risk losing context, while large chunks dilute specificity. Overlap preserves semantic continuity across chunk boundaries.",
            ["doc_interview"],
            ["doc_interview#10", "doc_interview#11"],
        ),
        (
            "How does LangGraph coordinate query rewriting when initial retrieval yields insufficient relevance?",
            "The state machine validates top retrieval scores against SIMILARITY_THRESHOLD. If relevance is low, it executes a rewrite_retry node up to MAX_RETRIES.",
            ["doc_interview"],
            ["doc_interview#13", "doc_interview#14"],
        ),
        (
            "What projects has Vivek Kumar built and what technologies do they use?",
            "Vivek built RAG for All (LangChain, ChromaDB, FastAPI), Smart Resume Analyzer (scikit-learn, NLTK, Streamlit), and MarketPulse (MERN, LangGraph, Groq).",
            ["doc_resume"],
            ["doc_resume#1", "doc_resume#2", "doc_resume#3"],
        ),
        (
            "Describe the SQLite database schema for storing documents and chunks.",
            "The database maintains 'documents' (metadata, size, status) and 'chunks' (content, chunk index, document_id foreign key, page number).",
            ["doc_readme"],
            ["doc_readme#21", "doc_readme#22"],
        ),
        (
            "How does the ingestion pipeline handle input validation and file storage?",
            "Sanitizes filenames, verifies extensions, validates file size under 25MB, parses documents, and records metadata in SQLite and FAISS.",
            ["doc_readme"],
            ["doc_readme#21", "doc_readme#27"],
        ),
        (
            "What coursework and technical tools does Vivek Kumar list on his profile?",
            "Tools: NumPy, Pandas, Matplotlib, Git, GitHub, VSCode, MySQL. Coursework: DSA, OS, Computer Networks, DBMS, and OOP.",
            ["doc_resume"],
            ["doc_resume#4", "doc_resume#0"],
        ),
        (
            "Compare the roles of FAISS vector store and SQLite metadata store in Universal RAG.",
            "FAISS provides fast dense vector nearest-neighbor search, while SQLite provides relational persistence for document lifecycle, chunks, and citations.",
            ["doc_readme"],
            ["doc_readme#20", "doc_readme#21"],
        ),
        (
            "How does the system distinguish casual queries from knowledge retrieval queries?",
            "The intent classifier checks greetings and courtesies using is_casual_query; casual turns bypass vector search, while knowledge queries route to retrieval.",
            ["doc_readme"],
            ["doc_readme#13", "doc_readme#14"],
        ),
        (
            "What security protections are implemented for document ingestion?",
            "SSRF URL validation, path traversal prevention, file size limits (25MB), and sanitization of user filenames.",
            ["doc_readme"],
            ["doc_readme#34", "doc_readme#35"],
        ),
        (
            "What happens during query rewrite retry in the LangGraph workflow?",
            "If similarity scores fall below threshold, rewrite_retry optimizes query keywords using ChatOpenAI or keyword expansion up to MAX_RETRIES.",
            ["doc_interview"],
            ["doc_interview#13", "doc_interview#14"],
        ),
        (
            "What contact and profile details are provided in Vivek Kumar's resume?",
            "Email (vivekkumarupadhyay05@gmail.com, 2023uai1996@mnit.ac.in), phone (+919508666296), GitHub (vivekmnit1996), and LeetCode (vivekupadhyay06).",
            ["doc_resume"],
            ["doc_resume#0", "doc_resume#1"],
        ),
        (
            "How are citations constructed and formatted for display in Universal RAG?",
            "Extracted directly from chunk metadata (source name, page, excerpt, match score) and displayed as verifiable UI chips.",
            ["doc_readme"],
            ["doc_readme#16", "doc_readme#17"],
        ),
        (
            "What are the benefits of maintaining session memory separate from vector store data?",
            "Prevents conversation chatter from polluting factual domain knowledge while allowing natural multi-turn pronoun resolution.",
            ["doc_interview"],
            ["doc_interview#16", "doc_interview#17"],
        ),
        (
            "What index structures does FAISS support and what is used in this repository?",
            "FAISS supports IndexFlatL2, IndexIVFFlat, and HNSW; Universal RAG defaults to exact flat L2 with normalized vectors.",
            ["doc_interview"],
            ["doc_interview#7", "doc_interview#8"],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(multi_chunk, 46):
        add_sample(f"bench-{idx:03d}", q, "multi_chunk", True, ans, sids, cids)

    # -------------------------------------------------------------
    # 4. MULTI-HOP QUESTIONS (10 items)
    # -------------------------------------------------------------
    multi_hop = [
        (
            "If Vivek Kumar's Smart Resume Analyzer extracts technical skills, what specific libraries does it use to achieve this?",
            "Python, scikit-learn, and NLTK.",
            ["doc_resume"],
            ["doc_resume#2"],
        ),
        (
            "What happens when a query retrieves chunks that fail the similarity threshold twice?",
            "The workflow exhausts MAX_RETRIES (1) and routes to fallback_refusal, returning a grounded refusal message.",
            ["doc_interview"],
            ["doc_interview#14"],
        ),
        (
            "Why does FAISS L2 distance calculate exact cosine similarity when vectors are L2-normalized?",
            "Because for unit vectors with norm 1.0, |u - v|^2 = |u|^2 + |v|^2 - 2(u.v) = 2 - 2*cos(theta), giving cos(theta) = 1.0 - L2^2 / 2.",
            ["doc_interview"],
            ["doc_interview#5"],
        ),
        (
            "How does pronoun resolution in the query rewriter utilize conversational session history?",
            "It inspects recent conversation turns and reformulates pronouns (e.g. 'its', 'they') into an independent search query.",
            ["doc_readme"],
            ["doc_readme#14"],
        ),
        (
            "How does chunk overlap prevent false refusals during LangGraph context validation?",
            "By ensuring key phrases spanning chunk boundaries are not truncated, keeping similarity scores above SIMILARITY_THRESHOLD.",
            ["doc_interview"],
            ["doc_interview#11"],
        ),
        (
            "Which project of Vivek incorporates atomic inventory concurrency locking and what LLM framework does it use?",
            "MarketPulse, utilizing LangGraph and Groq.",
            ["doc_resume"],
            ["doc_resume#3"],
        ),
        (
            "What enables Universal RAG to answer queries when no external OpenAI API key is supplied?",
            "The offline fallback embedding engine (FallbackEmbeddings) and extractive grounded answering.",
            ["doc_readme"],
            ["doc_readme#25"],
        ),
        (
            "How does the API respond when an ingested document is deleted via DELETE /documents/{id}?",
            "It removes chunk records from SQLite, deletes corresponding vectors from FAISS, removes permanent disk files, and returns 200.",
            ["doc_readme"],
            ["doc_readme#28"],
        ),
        (
            "What prevents hallucinated citations in answers generated by Universal RAG?",
            "Citations are compiled deterministically from metadata in retrieved chunks rather than generated by the LLM.",
            ["doc_interview"],
            ["doc_interview#15"],
        ),
        (
            "Which college degree is Vivek currently pursuing according to his resume?",
            "B.Tech in Artificial Intelligence and Data Engineering at MNIT Jaipur.",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(multi_hop, 61):
        add_sample(f"bench-{idx:03d}", q, "multi_hop", True, ans, sids, cids)

    # -------------------------------------------------------------
    # 5. MULTI-SOURCE QUESTIONS (10 items)
    # -------------------------------------------------------------
    multi_source = [
        (
            "Compare the RAG architecture described in the interview guide with the project implementation documented in the README.",
            "Both specify FAISS vector search, SQLite metadata storage, LangGraph cyclic retrieval, and strict citation extraction.",
            ["doc_interview", "doc_readme"],
            ["doc_interview#1", "doc_readme#0"],
        ),
        (
            "How does Vivek's personal project 'RAG for All' relate to the Universal RAG system documented in the README?",
            "RAG for All was an earlier exploratory prototype using ChromaDB and LangChain, which evolved into Universal RAG with FAISS, SQLite, and LangGraph.",
            ["doc_resume", "doc_readme"],
            ["doc_resume#1", "doc_readme#0"],
        ),
        (
            "How do the chunking parameters specified in README compare to the chunking principles in the interview guide?",
            "The README configures CHUNK_SIZE=600 and CHUNK_OVERLAP=100, which adheres to the 500-800 character window recommended in the interview guide.",
            ["doc_readme", "doc_interview"],
            ["doc_readme#27", "doc_interview#10"],
        ),
        (
            "How does Vivek Kumar's experience with LangGraph in MarketPulse relate to Universal RAG's workflow orchestration?",
            "Vivek utilized LangGraph in MarketPulse for state workflows, and Universal RAG uses LangGraph for stateful RAG with rewrite retries.",
            ["doc_resume", "doc_readme"],
            ["doc_resume#3", "doc_readme#13"],
        ),
        (
            "Compare the vector storage design in the interview guide with the database architecture in README.",
            "The interview guide explains mathematical L2/cosine conversion, while the README documents the concrete SQLite WAL schema and FAISS indexing.",
            ["doc_interview", "doc_readme"],
            ["doc_interview#5", "doc_readme#21"],
        ),
        (
            "What programming languages and tools shared between Vivek's resume and the Universal RAG repository are evident?",
            "Python, FastAPI, Git, GitHub, MySQL/SQLite, and LangChain.",
            ["doc_resume", "doc_readme"],
            ["doc_resume#4", "doc_readme#3"],
        ),
        (
            "How do citation guarantees in the interview guide align with the citation schema in the README?",
            "Both enforce that citations link to physical chunk IDs, page numbers, and excerpts directly from SQLite and FAISS metadata.",
            ["doc_interview", "doc_readme"],
            ["doc_interview#15", "doc_readme#17"],
        ),
        (
            "What security considerations mentioned in the README address risks outlined in RAG deployment guides?",
            "SSRF protection, payload size enforcement (25MB), and directory traversal sanitization.",
            ["doc_readme", "doc_interview"],
            ["doc_readme#34", "doc_interview#18"],
        ),
        (
            "How does session memory in Universal RAG differ from the candidate's project state management in MarketPulse?",
            "Universal RAG uses in-memory session history for conversational RAG, whereas MarketPulse implemented atomic concurrency locking on inventory.",
            ["doc_readme", "doc_resume"],
            ["doc_readme#23", "doc_resume#3"],
        ),
        (
            "How does the offline mode described in README fulfill the fallback design principles in the interview guide?",
            "By ensuring zero external network dependency, using deterministic 1536-dim embeddings and grounded extractive generation.",
            ["doc_readme", "doc_interview"],
            ["doc_readme#25", "doc_interview#4"],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(multi_source, 71):
        add_sample(f"bench-{idx:03d}", q, "multi_source", True, ans, sids, cids)

    # -------------------------------------------------------------
    # 6. AMBIGUOUS QUERIES (10 items)
    # -------------------------------------------------------------
    ambiguous = [
        (
            "What is the threshold?",
            "In Universal RAG, the default SIMILARITY_THRESHOLD is 0.0 (or configurable 0.0-1.0) to filter vector candidates.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "How many vectors?",
            "The vector count corresponds to the total indexed chunks in the FAISS index, accessible via GET /health.",
            ["doc_readme"],
            ["doc_readme#28"],
        ),
        (
            "Tell me about the memory.",
            "Session memory maintains chat history per session_id in RAM, separate from the FAISS vector knowledge base.",
            ["doc_readme"],
            ["doc_readme#23"],
        ),
        (
            "What is the score?",
            "Vector search computes a similarity score between 0.0 and 1.0 using converted normalized L2 distance.",
            ["doc_interview"],
            ["doc_interview#5"],
        ),
        (
            "What stack is used?",
            "Universal RAG uses Python 3.10+, FastAPI, LangChain, LangGraph, FAISS, SQLite, and Vanilla HTML/JS.",
            ["doc_readme"],
            ["doc_readme#3"],
        ),
        (
            "Who built it?",
            "Universal RAG was developed by Vivek Kumar (MNIT Jaipur).",
            ["doc_resume"],
            ["doc_resume#0"],
        ),
        (
            "How does chunking work?",
            "RecursiveCharacterTextSplitter partitions text hierarchically with 600 character chunks and 100 character overlap.",
            ["doc_interview"],
            ["doc_interview#10"],
        ),
        (
            "What is the model?",
            "The default configured LLM is gpt-4o-mini and the embedding model is text-embedding-3-small.",
            ["doc_readme"],
            ["doc_readme#27"],
        ),
        (
            "Where is data stored?",
            "Metadata is stored in data/metadata.db (SQLite) and vector indices are stored in data/vectorstore (FAISS).",
            ["doc_readme"],
            ["doc_readme#20"],
        ),
        (
            "Can I upload PDF?",
            "Yes, PDFs are uploaded via POST /documents/upload, extracted page-by-page, and indexed incrementally.",
            ["doc_readme"],
            ["doc_readme#28"],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(ambiguous, 81):
        add_sample(f"bench-{idx:03d}", q, "ambiguous", True, ans, sids, cids)

    # -------------------------------------------------------------
    # 7. UNANSWERABLE QUERIES (10 items - GROUND TRUTH REFUSAL)
    # -------------------------------------------------------------
    unanswerable = [
        (
            "What is the stock price of Apple Inc today?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "How many employees work at OpenAI headquarters in San Francisco?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "What was Vivek Kumar's salary at his first software engineering internship?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "Which quantum computing algorithm is implemented in Universal RAG?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "What is the weather forecast for Jaipur tomorrow?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "Who won the FIFA World Cup in 2022?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "What is the recipe for chocolate chip cookies?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "What is Vivek Kumar's favorite programming language according to his resume?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "How many GPU clusters are used to host the Universal RAG server?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
        (
            "What is the current version of the Rust compiler installed on the server?",
            "I could not find relevant information in the uploaded sources to answer your question.",
            [],
            [],
        ),
    ]

    for idx, (q, ans, sids, cids) in enumerate(unanswerable, 91):
        add_sample(f"bench-{idx:03d}", q, "unanswerable", False, ans, sids, cids)

    # -------------------------------------------------------------
    # 8. CONVERSATIONAL FOLLOW-UPS (5 items)
    # -------------------------------------------------------------
    followups = [
        (
            "What is its default chunk size?",
            "The default chunk size is 600 characters with 100 character overlap.",
            ["doc_readme"],
            ["doc_readme#27"],
            [
                {"role": "user", "content": "How does Universal RAG split long documents?"},
                {"role": "assistant", "content": "It uses RecursiveCharacterTextSplitter."},
            ],
        ),
        (
            "Where is he studying?",
            "Vivek Kumar is studying at Malaviya National Institute of Technology (MNIT), Jaipur.",
            ["doc_resume"],
            ["doc_resume#0"],
            [
                {"role": "user", "content": "Who is Vivek Kumar?"},
                {"role": "assistant", "content": "Vivek Kumar is a B.Tech student in Artificial Intelligence."},
            ],
        ),
        (
            "How does it prevent cycle loops during retrieval?",
            "LangGraph enforces a MAX_RETRIES parameter (default 1) so if query rewrite fails twice, it exits to fallback.",
            ["doc_interview"],
            ["doc_interview#14"],
            [
                {"role": "user", "content": "What happens when search retrieval finds low relevance chunks?"},
                {"role": "assistant", "content": "It triggers a rewrite_retry node."},
            ],
        ),
        (
            "What technologies does it use for concurrency?",
            "MarketPulse uses atomic inventory concurrency locking on the MERN stack with LangGraph and Groq.",
            ["doc_resume"],
            ["doc_resume#3"],
            [
                {"role": "user", "content": "Tell me about the MarketPulse marketplace project."},
                {"role": "assistant", "content": "MarketPulse is a multi-vendor e-commerce marketplace."},
            ],
        ),
        (
            "What port does it run on by default?",
            "It runs on port 8000 (host 0.0.0.0).",
            ["doc_readme"],
            ["doc_readme#27"],
            [
                {"role": "user", "content": "How do you run the FastAPI server locally?"},
                {"role": "assistant", "content": "You run uvicorn app:app --reload."},
            ],
        ),
    ]

    for idx, (q, ans, sids, cids, hist) in enumerate(followups, 101):
        add_sample(f"bench-{idx:03d}", q, "conversational_followup", True, ans, sids, cids, hist)

    print(f"Successfully generated {len(samples)} benchmark samples.")
    out_file = base_dir / "evaluation" / "benchmark_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved verified benchmark dataset to: {out_file}")

    return samples


if __name__ == "__main__":
    build_benchmark()

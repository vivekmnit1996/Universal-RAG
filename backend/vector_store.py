"""Persistent Vector Store layer with incremental indexing using FAISS."""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class FallbackEmbeddings(Embeddings):
    """
    Deterministic offline embedding model (1536 dimensions) for local testing and demonstration
    when an external OpenAI API key is not supplied.
    Generates consistent bag-of-words / n-gram unit vectors for offline search.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dimension
            val = (int(hashlib.sha256(word.encode()).hexdigest()[:4], 16) % 100) / 50.0 - 1.0
            vec[idx] += val

        # Character trigrams for morphological similarity
        for i in range(len(text) - 2):
            tri = text[i : i + 3].lower()
            idx = int(hashlib.sha1(tri.encode()).hexdigest(), 16) % self.dimension
            vec[idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings_model() -> Embeddings:
    """Return OpenAIEmbeddings if API key is configured, else fallback offline embeddings."""
    api_key = settings.OPENAI_API_KEY.strip()
    if api_key and not api_key.startswith("your-") and len(api_key) > 10:
        logger.info("Using OpenAIEmbeddings: %s", settings.EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=api_key,
        )
    logger.info("No active OpenAI API key detected. Using local FallbackEmbeddings (1536-dim).")
    return FallbackEmbeddings(dimension=1536)


class VectorStoreManager:
    """
    Manages persistent FAISS vector store with support for incremental indexing,
    incremental deletion, deduplication, and score thresholding.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.VECTOR_STORE_DIR
        self.embeddings: Embeddings = get_embeddings_model()
        self.vectorstore: Optional[FAISS] = None
        self.load()

    def load(self) -> None:
        """Load persisted FAISS index if index files exist on disk."""
        index_file = os.path.join(self.persist_dir, "index.faiss")
        pkl_file = os.path.join(self.persist_dir, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=self.persist_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    "Loaded existing FAISS index with %d vectors",
                    self.vectorstore.index.ntotal,
                )
            except Exception as e:
                logger.warning("Could not load existing FAISS index: %s. Re-initializing.", e)
                self.vectorstore = None
        else:
            self.vectorstore = None

    def save(self) -> None:
        """Persist current FAISS index to disk."""
        if self.vectorstore is not None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vectorstore.save_local(self.persist_dir)
            logger.info("Saved FAISS index to %s", self.persist_dir)

    def add_chunks(self, chunks: List[Document]) -> int:
        """
        Incrementally add newly created chunks to the vector store.
        Embeds only the new chunks without re-processing existing documents.
        """
        if not chunks:
            return 0

        ids = [c.metadata.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings, ids=ids)
        else:
            self.vectorstore.add_documents(chunks, ids=ids)

        self.save()
        total = self.vectorstore.index.ntotal
        logger.info("Added %d new chunks. Total index vectors: %d", len(chunks), total)
        return len(chunks)

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Incrementally remove chunks by ID from the vector store.
        """
        if not chunk_ids or self.vectorstore is None:
            return 0

        # Filter only existing IDs in the index
        docstore = self.vectorstore.docstore
        existing_ids = [cid for cid in chunk_ids if cid in docstore._dict]

        if not existing_ids:
            return 0

        self.vectorstore.delete(existing_ids)

        # Check if index is now empty
        if self.vectorstore.index.ntotal == 0:
            self.reset()
        else:
            self.save()

        logger.info("Deleted %d chunks from vector store.", len(existing_ids))
        return len(existing_ids)

    def reset(self) -> None:
        """Completely reset vector store and clear disk files."""
        self.vectorstore = None
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
        logger.info("Vector store reset and emptied.")

    def get_total_vectors(self) -> int:
        """Return total number of vectors in FAISS index."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal

    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform vector similarity search, normalize distance into a 0.0-1.0 similarity score,
        apply threshold filtering, and deduplicate overlapping chunks.
        """
        if self.vectorstore is None or self.vectorstore.index.ntotal == 0:
            return []

        top_k = k or settings.TOP_K
        sim_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

        # Query more candidates initially to allow clean deduplication
        candidate_k = min(top_k * 2, self.vectorstore.index.ntotal)
        raw_results = self.vectorstore.similarity_search_with_score(query, k=candidate_k)

        scored_results: List[Tuple[Document, float]] = []
        seen_contents = set()

        for doc, distance in raw_results:
            # For normalized unit embeddings, cosine similarity s = 1.0 - (L2_dist^2 / 2.0)
            l2_sq = float(distance) ** 2
            similarity = max(0.0, min(1.0, 1.0 - (l2_sq / 2.0)))

            if similarity < sim_threshold:
                continue

            # Deduplication: normalize content snippet to detect exact or near-duplicate chunks
            content_key = " ".join(doc.page_content.split()[:20]).lower()
            if content_key in seen_contents:
                continue
            seen_contents.add(content_key)

            scored_results.append((doc, round(similarity, 4)))

            if len(scored_results) >= top_k:
                break

        return scored_results

"""Source citation extractor and formatter for grounded RAG generation."""

from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document


def build_citations(retrieved_items: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    """
    Extract grounded source citations directly from retrieved chunk metadata.
    Guarantees no hallucinated sources by building strictly from retrieved chunks.
    """
    citations: List[Dict[str, Any]] = []
    seen = set()

    for doc, score in retrieved_items:
        meta = doc.metadata
        source_name = meta.get("source_name", "Unknown Source")
        source_type = meta.get("source_type", "document")
        page_num = meta.get("page_number")
        row_start = meta.get("row_start")
        row_end = meta.get("row_end")
        chunk_id = meta.get("chunk_id", "")
        source_uri = meta.get("source_uri", "")

        # Uniqueness key to avoid identical citation entries
        unique_key = (source_name, page_num, row_start, row_end)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        # Excerpt for provenance
        text = doc.page_content.strip()
        excerpt = text[:140] + "..." if len(text) > 140 else text

        # Formatted citation label
        if source_type == "pdf" and page_num is not None:
            label = f"{source_name} — Page {page_num}"
        elif source_type == "csv" and row_start is not None:
            label = f"{source_name} — Rows {row_start}-{row_end}"
        elif source_type == "youtube":
            video_id = meta.get("video_id", "")
            label = f"YouTube ({video_id}) — Video Transcript"
        elif source_type == "web":
            title = meta.get("title") or source_name
            label = f"{title} — Web Source"
        else:
            label = f"{source_name}"

        citations.append(
            {
                "label": label,
                "source_name": source_name,
                "source_type": source_type,
                "source_uri": source_uri,
                "page": page_num,
                "chunk_id": chunk_id,
                "score": score,
                "excerpt": excerpt,
            }
        )

    return citations


def format_citations_text(citations: List[Dict[str, Any]]) -> str:
    """Format structured citations into a clean, interview-defensible text block."""
    if not citations:
        return "No sources available."

    lines = ["\n\nSources:"]
    for c in citations:
        lines.append(f"- {c['label']}")
    return "\n".join(lines)

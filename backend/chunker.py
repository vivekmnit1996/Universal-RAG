"""Chunking engine for splitting parsed documents into metadata-enriched chunks."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import settings
from backend.document_parser import ParsedDocument


def create_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Instantiate a RecursiveCharacterTextSplitter with configured chunk boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document_batch(
    parsed_docs: List[ParsedDocument],
    document_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of ParsedDocument objects into LangChain Document chunks.
    Attaches comprehensive chunk-level metadata:
      - document_id
      - chunk_id: '{document_id}#{chunk_index}'
      - chunk_index
      - source_name
      - source_type
      - source_uri
      - page_number (if applicable)
      - total_chunks
    """
    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Document] = []
    chunk_index = 0

    for parsed_doc in parsed_docs:
        raw_text = parsed_doc.content.strip()
        if not raw_text:
            continue

        text_pieces = splitter.split_text(raw_text)

        for piece in text_pieces:
            piece_clean = piece.strip()
            if not piece_clean:
                continue

            chunk_id = f"{document_id}#{chunk_index}"
            chunk_metadata = dict(parsed_doc.metadata)
            chunk_metadata.update(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            )

            all_chunks.append(
                Document(
                    page_content=piece_clean,
                    metadata=chunk_metadata,
                )
            )
            chunk_index += 1

    # Attach total_chunks to all chunk metadata for context awareness
    total = len(all_chunks)
    for c in all_chunks:
        c.metadata["total_chunks"] = total

    return all_chunks

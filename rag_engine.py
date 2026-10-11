"""Compatibility adapter bridging legacy rag_engine.py interface to the new backend architecture."""

from typing import Any, List, Tuple
from langchain_core.documents import Document
from backend.chunker import chunk_document_batch
from backend.document_parser import (
    detect_source_type,
    parse_source,
    ParsedDocument,
)
from backend.rag_workflow import RAGWorkflow, get_llm
from backend.vector_store import VectorStoreManager


def parse_input(user_input: str) -> List[Document]:
    """Parse raw input source and return LangChain Documents."""
    parsed_docs = parse_source(user_input)
    chunks = chunk_document_batch(parsed_docs, document_id="legacy_doc")
    return chunks


def build_rag(documents: List[Document]) -> Tuple[Any, Any]:
    """Build persistent vector store and LLM from document chunks."""
    vm = VectorStoreManager()
    if documents:
        vm.add_chunks(documents)
    llm = get_llm()
    return vm, llm


def ask_rag(question: str, retriever: Any, llm: Any = None) -> str:
    """Execute grounded RAG workflow over retriever and return answer string."""
    if isinstance(retriever, VectorStoreManager):
        vm = retriever
    else:
        vm = VectorStoreManager()

    workflow = RAGWorkflow(vector_manager=vm, llm=llm)
    result = workflow.run(question)
    return result["answer"]

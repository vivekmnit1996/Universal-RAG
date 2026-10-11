"""LangGraph Stateful Workflow for Multi-Turn Conversational RAG.

Separates Chat Memory from RAG Vector Knowledge:
- START -> load_conversation -> classify_query
  -> CASUAL: direct_response -> update_memory -> END
  -> KNOWLEDGE: rewrite_query -> retrieve_docs -> validate_context
       -> RELEVANT: generate_answer -> format_citations -> update_memory -> END
       -> IRRELEVANT (retry < max_retries): rewrite_retry -> retrieve_docs -> validate_context ...
       -> IRRELEVANT (retry >= max_retries): fallback_refusal -> update_memory -> END
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from backend.citations import build_citations
from backend.config import settings
from backend.session_memory import session_memory
from backend.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# CASUAL QUERY DETECTION CONSTANTS
# -------------------------------------------------------------
GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy", "heya", "howdy", "sup",
    "greetings", "good morning", "good afternoon", "good evening", "good day",
}

COURTESY_WORDS = {
    "thanks", "thank you", "thx", "ty", "thank u", "okay", "ok", "k",
    "cool", "great", "nice", "awesome", "got it", "understood", "sure",
    "alright", "perfect",
}

FAREWELL_WORDS = {
    "bye", "goodbye", "see ya", "cya", "farewell", "later", "take care", "good night",
}

IDENTITY_PHRASES = {
    "who are you", "what can you do", "what are you", "what is your name",
    "tell me about yourself", "help",
}


def is_casual_query(query: str) -> Tuple[bool, str]:
    """
    Lightweight classification to check if a query is a greeting, courtesy, or casual conversational turn.
    Returns (is_casual: bool, category: str).
    """
    clean_q = re.sub(r"[^\w\s]", "", query.strip().lower())
    words = clean_q.split()

    if not words:
        return True, "empty"

    # Check courtesies (e.g. "thanks", "thank you so much")
    if any(c in clean_q for c in ["thank you", "thanks", "thx", "thank u"]) and len(words) <= 6:
        return True, "courtesy"
    if clean_q in COURTESY_WORDS or (len(words) <= 2 and words[0] in COURTESY_WORDS):
        return True, "courtesy"

    # Check greetings (e.g. "hi", "hello there", "good morning")
    if any(clean_q.startswith(g) for g in GREETING_WORDS) and len(words) <= 4:
        return True, "greeting"
    if clean_q in GREETING_WORDS:
        return True, "greeting"

    # Check farewells (e.g. "bye", "goodbye")
    if any(clean_q.startswith(f) for f in FAREWELL_WORDS) and len(words) <= 4:
        return True, "farewell"

    # Identity / Help phrases
    if any(phrase in clean_q for phrase in IDENTITY_PHRASES) and len(words) <= 5:
        return True, "identity"

    # Resume Evaluation queries
    if bool(re.search(r"\b(rate|score|evaluate|review|grade|assess)\b.*\b(resume|cv)\b", clean_q)) or clean_q in {
        "rate my resume", "rate resume", "evaluate my resume", "review my resume", "score my resume", "check my resume"
    }:
        return True, "resume_eval"

    return False, "knowledge"


# -------------------------------------------------------------
# GRAPH STATE SCHEMA
# -------------------------------------------------------------
class RAGState(TypedDict):
    """Typed state schema for the LangGraph Conversational RAG pipeline."""

    question: str
    session_id: str
    chat_history: List[BaseMessage]
    intent: str  # 'casual' vs 'knowledge'
    casual_category: str
    standalone_query: str
    retrieved_docs: List[Tuple[Document, float]]
    is_relevant: bool
    retry_count: int
    max_retries: int
    context_text: str
    answer: str
    citations: List[Dict[str, Any]]
    error: Optional[str]


def get_llm():
    """Return ChatOpenAI instance if API key is configured, else None."""
    api_key = settings.OPENAI_API_KEY.strip()
    if api_key and not api_key.startswith("your-") and len(api_key) > 10:
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            openai_api_key=api_key,
        )
    return None


class RAGWorkflow:
    """
    Stateful conversational workflow orchestrating intent classification,
    standalone query rewriting, threshold-enforced vector retrieval,
    grounded generation, and separate session memory updates.
    """

    def __init__(self, vector_manager: VectorStoreManager, llm: Optional[Any] = None):
        self.vector_manager = vector_manager
        self.llm = llm if llm is not None else get_llm()
        self.app = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RAGState)

        # Nodes
        builder.add_node("load_conversation", self._node_load_conversation)
        builder.add_node("classify_query", self._node_classify_query)
        builder.add_node("direct_response", self._node_direct_response)
        builder.add_node("rewrite_query", self._node_rewrite_query)
        builder.add_node("rewrite_retry", self._node_rewrite_retry)
        builder.add_node("retrieve_docs", self._node_retrieve_docs)
        builder.add_node("validate_context", self._node_validate_context)
        builder.add_node("fallback_refusal", self._node_fallback_refusal)
        builder.add_node("generate_answer", self._node_generate_answer)
        builder.add_node("format_citations", self._node_format_citations)
        builder.add_node("update_memory", self._node_update_memory)

        # Flow edges
        builder.add_edge(START, "load_conversation")
        builder.add_edge("load_conversation", "classify_query")

        # Branch based on intent
        builder.add_conditional_edges(
            "classify_query",
            self._route_after_classification,
            {
                "casual": "direct_response",
                "knowledge": "rewrite_query",
            },
        )

        builder.add_edge("direct_response", "update_memory")

        builder.add_edge("rewrite_query", "retrieve_docs")
        builder.add_edge("rewrite_retry", "retrieve_docs")
        builder.add_edge("retrieve_docs", "validate_context")

        # Branch based on context relevance
        builder.add_conditional_edges(
            "validate_context",
            self._route_after_validation,
            {
                "generate": "generate_answer",
                "retry": "rewrite_retry",
                "fallback": "fallback_refusal",
            },
        )

        builder.add_edge("fallback_refusal", "update_memory")
        builder.add_edge("generate_answer", "format_citations")
        builder.add_edge("format_citations", "update_memory")
        builder.add_edge("update_memory", END)

        return builder.compile()

    # -------------------------------------------------------------
    # NODE IMPLEMENTATIONS
    # -------------------------------------------------------------
    def _node_load_conversation(self, state: RAGState) -> Dict[str, Any]:
        """Fetch isolated conversation history for the given session_id from SessionMemoryManager."""
        session_id = state.get("session_id", "default")
        history = session_memory.get_history(session_id)
        cleaned_question = state["question"].strip()

        logger.info(
            "[Session: %s] Loaded %d prior messages from session memory",
            session_id,
            len(history),
        )

        return {
            "session_id": session_id,
            "question": cleaned_question,
            "chat_history": history,
            "retry_count": 0,
            "max_retries": settings.MAX_RETRIES,
            "retrieved_docs": [],
            "citations": [],
            "is_relevant": False,
            "context_text": "",
        }

    def _node_classify_query(self, state: RAGState) -> Dict[str, Any]:
        """Classify user query into casual chat vs. knowledge retrieval."""
        q = state["question"]
        is_casual, category = is_casual_query(q)

        logger.info(
            "[Intent Classifier] Query: '%s' -> intent: %s (category: %s)",
            q,
            "casual" if is_casual else "knowledge",
            category,
        )

        return {
            "intent": "casual" if is_casual else "knowledge",
            "casual_category": category,
        }

    def _route_after_classification(self, state: RAGState) -> str:
        """Route query: casual bypasses vector search completely."""
        return "casual" if state["intent"] == "casual" else "knowledge"

    def _node_direct_response(self, state: RAGState) -> Dict[str, Any]:
        """Generate conversational response for greetings/courtesies with ZERO retrieval or citations."""
        cat = state.get("casual_category", "greeting")

        if cat == "greeting":
            answer = "Hello! I am Universal RAG. How can I help you today? You can ask questions grounded in your uploaded documents or ingested sources."
        elif cat == "courtesy":
            answer = "You're welcome! Feel free to ask any other questions about your documents."
        elif cat == "farewell":
            answer = "Goodbye! Let me know if you need any assistance in the future."
        elif cat == "identity":
            answer = "I am Universal RAG, an AI assistant designed to answer questions strictly grounded in your uploaded documents (PDFs, CSVs, Web URLs, YouTube transcripts, and text) with verifiable citations."
        elif cat == "resume_eval":
            try:
                from backend.service import service
                eval_res = service.evaluate_resume()
                score = eval_res["overall_score"]
                band = eval_res["rating_band"]
                doc_name = eval_res["source_name"]
                strengths = "\n".join(f"• {s}" for s in eval_res.get("strengths", [])[:3])
                improvements = "\n".join(
                    f"• **{imp['category']}**: {imp['recommendation']}"
                    for imp in eval_res.get("prioritized_improvements", [])[:3]
                )
                rubric_lines = "\n".join(
                    f"• **{c.replace('_', ' ').title()}**: {info['score']}/{info['weight']} pts"
                    for c, info in eval_res.get("criterion_scores", {}).items()
                )

                answer = (
                    f"### ⭐ Resume Evaluation Report: {doc_name}\n\n"
                    f"• **Overall ATS Score:** **{score}/100** — *{band}*\n\n"
                    f"**Rubric Breakdown:**\n{rubric_lines}\n\n"
                    f"**Key Strengths:**\n{strengths}\n\n"
                    f"**Prioritized Recommendations:**\n{improvements}"
                )
            except Exception as e:
                answer = (
                    "To evaluate your resume, please upload your resume file (PDF) first using the "
                    f"sidebar upload tab. ({str(e)})"
                )
        else:
            answer = "Hello! Please ask a question related to your uploaded documents or data sources."

        logger.info("[Direct Response] Handled casual turn without vector retrieval.")

        return {
            "answer": answer,
            "standalone_query": state["question"],
            "retrieved_docs": [],
            "citations": [],
            "context_text": "",
            "is_relevant": False,
        }

    def _node_rewrite_query(self, state: RAGState) -> Dict[str, Any]:
        """
        Initial standalone query generator for knowledge search.
        Resolves pronouns (he/she/it/they) and follow-up references using chat history.
        """
        question = state["question"]
        history = state.get("chat_history", [])

        # If no history, question is already standalone
        if not history:
            return {"standalone_query": question, "retry_count": 0}

        reference_pattern = re.compile(
            r"\b(it|its|they|them|their|this|that|these|those|the former|the latter|he|him|his|she|her|who|which|what about|how about|tell me more|what else|candidate|resume|applicant|person|score|rating|feedback|evaluation|project|projects|skills|experience|education|previous|earlier|before)\b",
            re.IGNORECASE,
        )
        has_reference = bool(reference_pattern.search(question))
        is_short_followup = len(question.split()) <= 7

        standalone = question

        if self.llm is not None and (has_reference or is_short_followup):
            try:
                history_lines = []
                for msg in history[-6:]:
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    history_lines.append(f"{role}: {msg.content[:250]}")
                history_text = "\n".join(history_lines)

                prompt = (
                    "You are a search query reformulator for a document Q&A system. "
                    "Given the recent conversation history and a follow-up user question, "
                    "rewrite the follow-up question into an independent, standalone search query containing all relevant entities, names, and context. "
                    "If the question is already fully self-contained, return it as-is. "
                    "Do NOT answer the question. Only return the standalone query text.\n\n"
                    f"Conversation History:\n{history_text}\n\n"
                    f"Follow-up Question: {question}\n\n"
                    "Standalone Query:"
                )
                response = self.llm.invoke(prompt)
                rewritten = response.content.strip().replace('"', "")
                if rewritten:
                    standalone = rewritten
                    logger.info("[Query Rewriter] Rewrote '%s' -> '%s'", question, standalone)
            except Exception as e:
                logger.warning("[Query Rewriter] LLM call failed: %s", e)
                standalone = question
        elif (has_reference or is_short_followup) and history:
            # Deterministic heuristic resolution if offline
            last_user_msg = None
            for msg in reversed(history):
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg.content
                    break
            if last_user_msg:
                prior_keywords = [
                    w for w in last_user_msg.split()
                    if len(w) > 3 and w.lower() not in {"what", "how", "why", "when", "tell", "show", "give"}
                ]
                if prior_keywords:
                    context_keyword = " ".join(prior_keywords[:3])
                    standalone = f"{question} (context: {context_keyword})"
                    logger.info("[Heuristic Rewriter] Appended context: '%s'", standalone)

        return {"standalone_query": standalone, "retry_count": 0}

    def _node_rewrite_retry(self, state: RAGState) -> Dict[str, Any]:
        """Reformulate search query when initial retrieval returned low relevance."""
        retries = state.get("retry_count", 0) + 1
        current_query = state.get("standalone_query", state["question"])

        rewritten = current_query
        if self.llm is not None:
            try:
                prompt = (
                    "You are optimizing a search query for a vector database. "
                    "Rephrase the following question using clear, alternative technical keywords:\n"
                    f"Original Question: {current_query}\n"
                    "Optimized Search Query:"
                )
                response = self.llm.invoke(prompt)
                candidate = response.content.strip().replace('"', "")
                if candidate:
                    rewritten = candidate
                    logger.info("[Retry Rewriter] Optimized query: '%s'", rewritten)
            except Exception as e:
                logger.warning("[Retry Rewriter] LLM call failed: %s", e)
                rewritten = current_query
        else:
            # Keyword expansion
            words = [w for w in current_query.split() if len(w) > 2]
            rewritten = " ".join(words)

        return {
            "retry_count": retries,
            "standalone_query": rewritten,
        }

    def _node_retrieve_docs(self, state: RAGState) -> Dict[str, Any]:
        """Execute vector similarity search using the standalone query against FAISS."""
        query = state.get("standalone_query", state["question"])
        logger.info(
            "[Retriever] Searching FAISS for: '%s' (threshold: %.3f)",
            query,
            settings.SIMILARITY_THRESHOLD,
        )

        results = self.vector_manager.similarity_search_with_scores(
            query=query,
            k=settings.TOP_K,
            threshold=settings.SIMILARITY_THRESHOLD,
        )

        logger.info("[Retriever] Found %d chunks above threshold", len(results))
        return {"retrieved_docs": results}

    def _node_validate_context(self, state: RAGState) -> Dict[str, Any]:
        """Check if retrieved chunks exist and meet relevance threshold."""
        docs = state.get("retrieved_docs", [])
        if not docs:
            logger.info("[Validation] Zero chunks retrieved above threshold.")
            return {"is_relevant": False}

        top_score = max(score for _, score in docs)
        is_rel = top_score >= settings.SIMILARITY_THRESHOLD
        logger.info(
            "[Validation] Top similarity score: %.3f (Threshold: %.3f) -> relevant: %s",
            top_score,
            settings.SIMILARITY_THRESHOLD,
            is_rel,
        )
        return {"is_relevant": is_rel}

    def _route_after_validation(self, state: RAGState) -> str:
        """Route to generator if relevant; retry or conversational routing if irrelevant."""
        if state["is_relevant"]:
            return "generate"

        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.MAX_RETRIES)

        if retry_count < max_retries:
            return "retry"

        # Check if query is conversational follow-up that can be answered from chat history
        history = state.get("chat_history", [])
        q = state["question"].lower()
        conv_cues = ["what did i ask", "what was my", "what did you say", "repeat", "summarize our", "previous", "earlier", "we were talking"]
        if history and (any(cue in q for cue in conv_cues) or len(state["question"].split()) <= 4):
            logger.info("[Validation] Routing follow-up to generator using chat history.")
            return "generate"

        return "fallback"

    def _node_fallback_refusal(self, state: RAGState) -> Dict[str, Any]:
        """Return clear refusal when retrieved chunks are absent or below relevance threshold."""
        logger.info("[Fallback] No relevant knowledge found. Returning refusal with 0 sources.")
        return {
            "answer": "I could not find relevant information in the uploaded sources to answer your question.",
            "citations": [],
            "context_text": "",
        }

    def _node_generate_answer(self, state: RAGState) -> Dict[str, Any]:
        """Generate answer grounded in retrieved chunks and conversational history."""
        docs = state.get("retrieved_docs", [])
        history = state.get("chat_history", [])

        if not docs and not history:
            return {
                "answer": "I could not find relevant information in the uploaded sources to answer your question.",
                "context_text": "",
            }

        # Format retrieved context
        context_parts = []
        for i, (doc, _) in enumerate(docs, 1):
            source = doc.metadata.get("source_name", "Unknown")
            page = doc.metadata.get("page_number")
            loc = f"Page {page}" if page else f"Chunk {i}"
            context_parts.append(f"[{source} — {loc}]:\n{doc.page_content}")

        context_text = "\n\n".join(context_parts)

        if self.llm is not None:
            # Include recent chat history context for conversational fluidity (up to 10 messages)
            history_lines = []
            for msg in history[-10:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_lines.append(f"{role}: {msg.content}")
            history_ctx = "\n".join(history_lines) if history_lines else "None"

            prompt = (
                "You are a helpful, knowledgeable, and grounded AI assistant. "
                "You answer questions, synthesize information, evaluate profiles/resumes, and converse naturally over uploaded documents.\n\n"
                "Operating Principles:\n"
                "1. Factual Grounding: Ground all specific facts, qualifications, numbers, and statements strictly in the Retrieved Context. Do NOT fabricate companies, degrees, or metrics not present in the sources.\n"
                "2. Constructive Analysis & Evaluation: When asked to evaluate, rate, review, summarize, or analyze (e.g. assessing a candidate's resume, strengths, weaknesses, technical proficiency, or project impact), synthesize the context constructively. Provide clear ratings/scores, explain the reasoning, highlight strong points, and offer actionable suggestions.\n"
                "3. Conversational Memory: Seamlessly maintain context from the Recent Conversation. If the user refers to previous answers, entities, or questions, respond naturally without forgetting earlier context.\n"
                "4. Honest Refusal: ONLY state 'I could not find relevant information in the uploaded sources to answer your question.' if the uploaded context contains absolutely no information relevant to the user's topic and the question cannot be answered from the conversation history.\n\n"
                f"Recent Conversation:\n{history_ctx}\n\n"
                f"Retrieved Context:\n{context_text if context_text else 'No new documents retrieved; refer to recent conversation history.'}\n\n"
                f"User Question:\n{state['question']}\n\n"
                "Helpful Answer:"
            )
            try:
                response = self.llm.invoke(prompt)
                answer = response.content.strip()
            except Exception as e:
                logger.error("[Generator] LLM call failed: %s", e)
                answer = f"Error during answer generation: {str(e)}"
        else:
            # Deterministic grounded offline response for testing and evaluation
            if docs:
                q_words = set(state["question"].lower().split())
                best_sentences = []
                for doc, _ in docs:
                    for sent in doc.page_content.split("."):
                        sent_clean = sent.strip()
                        if len(sent_clean) > 20 and any(w in sent_clean.lower() for w in q_words):
                            best_sentences.append(sent_clean)

                if best_sentences:
                    extracted = ". ".join(best_sentences[:2]) + "."
                    answer = f"According to the source ({docs[0][0].metadata.get('source_name', 'document')}): {extracted}"
                else:
                    answer = (
                        f"Based on retrieved context from {docs[0][0].metadata.get('source_name', 'source')}: "
                        f"{docs[0][0].page_content[:250]}."
                    )
            elif history:
                answer = f"Based on our earlier conversation: You asked about {history[-1].content[:60]}."
            else:
                answer = "I could not find relevant information in the uploaded sources to answer your question."

        return {"answer": answer, "context_text": context_text}

    def _node_format_citations(self, state: RAGState) -> Dict[str, Any]:
        """Extract verified source citations strictly from retrieved RAG chunks when answer is grounded."""
        ans_lower = state.get("answer", "").lower()
        refusal_signals = [
            "not present in the sources",
            "couldn't find",
            "could not find",
            "not available in the",
            "not mentioned in the",
            "does not contain",
            "do not contain",
            "no relevant information",
        ]
        if any(sig in ans_lower for sig in refusal_signals):
            logger.info("[Citations] Answer indicated refusal. Suppressing citation chips.")
            return {"citations": []}

        docs = state.get("retrieved_docs", [])
        citations = build_citations(docs)
        return {"citations": citations}

    def _node_update_memory(self, state: RAGState) -> Dict[str, Any]:
        """
        Commit the completed turn to SessionMemoryManager.
        Keeps chat history isolated from RAG vector knowledge.
        """
        session_id = state.get("session_id", "default")
        question = state["question"]
        answer = state.get("answer", "")

        if question and answer:
            session_memory.add_turn(session_id, question, answer)
            logger.info("[Memory] Saved turn to session '%s'", session_id)

        return {}

    def run(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Execute the conversational LangGraph workflow."""
        initial_state: RAGState = {
            "question": question,
            "session_id": session_id,
            "chat_history": [],
            "intent": "knowledge",
            "casual_category": "",
            "standalone_query": question,
            "retrieved_docs": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": settings.MAX_RETRIES,
            "context_text": "",
            "answer": "",
            "citations": [],
            "error": None,
        }
        final_state = self.app.invoke(initial_state, config={"recursion_limit": 15})
        return {
            "question": final_state["question"],
            "session_id": final_state.get("session_id", session_id),
            "intent": final_state.get("intent", "knowledge"),
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "standalone_query": final_state.get("standalone_query"),
            "sources_count": len(final_state.get("retrieved_docs", [])),
        }

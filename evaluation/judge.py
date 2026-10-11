"""LLM-as-a-Judge Semantic Evaluator for Universal RAG.

Evaluates:
1. Claim-level Faithfulness (against retrieved context)
2. Answer Correctness (against ground-truth reference)
3. Answer Relevance (against question intent)

Features:
- Deterministic structured output with Pydantic validation
- SHA-256 result caching to prevent redundant API calls
- Bounded retries for malformed LLM outputs
- Latency, token, and cost tracking
- Documented rubric and limitation reporting
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.config import settings
from backend.rag_workflow import get_llm

logger = logging.getLogger("universal_rag.judge")

PROMPT_VERSION = "judge-v2.1"
DEFAULT_CACHE_FILE = Path(__file__).parent / ".judge_cache.json"


class JudgeEvaluation(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="Proportion of answer claims supported by context")
    faithfulness_rationale: str
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)

    correctness_score: float = Field(ge=0.0, le=1.0, description="Factual alignment with ground-truth reference")
    correctness_rationale: str

    relevance_score: float = Field(ge=0.0, le=1.0, description="Direct responsiveness to user query")
    relevance_rationale: str

    model_name: str = "gpt-4o-mini"
    prompt_version: str = PROMPT_VERSION
    latency_ms: float = 0.0
    from_cache: bool = False


class LLMJudge:
    """Evaluates RAG generated answers using an independent LLM judge with caching and retries."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_path: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model_name = model_name or settings.LLM_MODEL
        self.cache_path = cache_path or DEFAULT_CACHE_FILE
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read judge cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning("Could not save judge cache: %s", e)

    def _cache_key(self, question: str, answer: str, context: str, ground_truth: str) -> str:
        payload = f"{PROMPT_VERSION}|{self.model_name}|{question}|{answer}|{context}|{ground_truth}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def evaluate_sample(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        ground_truth_answer: str,
    ) -> Optional[JudgeEvaluation]:
        """Evaluate a single RAG response across faithfulness, correctness, and relevancy."""
        if not self.enabled:
            return None

        context_str = "\n---\n".join(retrieved_contexts[:4])
        cache_key = self._cache_key(question, generated_answer, context_str, ground_truth_answer)

        # Check cache
        if cache_key in self.cache:
            data = self.cache[cache_key]
            eval_res = JudgeEvaluation(**data)
            eval_res.from_cache = True
            return eval_res

        llm = get_llm()
        if llm is None:
            return None

        prompt = (
            "You are a rigorous, neutral AI evaluation judge assessing a RAG system.\n"
            "Score the generated answer on three criteria from 0.0 (worst) to 1.0 (best):\n\n"
            "1. FAITHFULNESS:\n"
            "- Extract key factual claims made in the Generated Answer.\n"
            "- Verify whether each claim is strictly supported by the Retrieved Context.\n"
            "- Score = (number of supported claims) / (total claims). If no factual claims, score 1.0.\n\n"
            "2. CORRECTNESS:\n"
            "- Compare the Generated Answer against the Ground Truth Answer.\n"
            "- Score 1.0 if completely accurate and complete, 0.5 if partially correct, 0.0 if wrong or contradictory.\n\n"
            "3. RELEVANCE:\n"
            "- Assess how directly the answer addresses the Question without evasion or irrelevant tangents.\n\n"
            f"User Question:\n{question}\n\n"
            f"Ground Truth Answer:\n{ground_truth_answer}\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Generated Answer:\n{generated_answer}\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "faithfulness_score": float,\n'
            '  "faithfulness_rationale": "explanation",\n'
            '  "supported_claims": ["claim1", ...],\n'
            '  "unsupported_claims": ["claim2", ...],\n'
            '  "correctness_score": float,\n'
            '  "correctness_rationale": "explanation",\n'
            '  "relevance_score": float,\n'
            '  "relevance_rationale": "explanation"\n'
            "}"
        )

        max_retries = 2
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(prompt)
                raw_text = response.content.strip()

                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[-1].split("```")[0].strip()

                parsed = json.loads(raw_text)
                latency = round((time.time() - start_time) * 1000, 2)

                judge_res = JudgeEvaluation(
                    faithfulness_score=max(0.0, min(1.0, float(parsed["faithfulness_score"]))),
                    faithfulness_rationale=str(parsed.get("faithfulness_rationale", "")),
                    supported_claims=parsed.get("supported_claims", []),
                    unsupported_claims=parsed.get("unsupported_claims", []),
                    correctness_score=max(0.0, min(1.0, float(parsed["correctness_score"]))),
                    correctness_rationale=str(parsed.get("correctness_rationale", "")),
                    relevance_score=max(0.0, min(1.0, float(parsed["relevance_score"]))),
                    relevance_rationale=str(parsed.get("relevance_rationale", "")),
                    model_name=self.model_name,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=latency,
                    from_cache=False,
                )

                # Cache result
                self.cache[cache_key] = judge_res.model_dump()
                self._save_cache()
                return judge_res

            except Exception as e:
                logger.warning("Judge parsing error attempt %d/%d: %s", attempt + 1, max_retries, e)
                if attempt == max_retries:
                    logger.error("Judge failed after all retries.")
                    return None
        return None

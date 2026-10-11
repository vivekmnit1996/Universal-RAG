"""Tests for Resume Evaluation Engine and REST API Integration."""

import pytest
from fastapi.testclient import TestClient

from app import app
from backend.resume_evaluator import (
    RUBRIC_WEIGHTS,
    ResumeParser,
    evaluate_resume,
    evaluate_resume_deterministically,
    get_rating_band,
)
from backend.service import service

client = TestClient(app)

SAMPLE_RESUME_TEXT = """
Vivek Kumar
B.Tech in Artificial Intelligence and Data Engineering | Malaviya National Institute of Technology, Jaipur
Roll No: 2023UAI1996 | +919508666296 | vivekkumarupadhyay05@gmail.com | github.com/vivekmnit1996 | leetcode.com/vivekupadhyay06

Education
Malaviya National Institute of Technology, Jaipur (2023–Present)
B.Tech in Artificial Intelligence and Data Engineering, CGPA: 7.85 / 10.0
CHS, BHU, Varanasi (2020)
10th Grade CBSE Board Percentage: 97.8%

Technical Skills
Languages: Python, C++, SQL, JavaScript
Frameworks & Libraries: FastAPI, Flask, React, LangChain, LangGraph, scikit-learn, NLTK, pandas, numpy
Databases: MySQL, SQLite, FAISS, ChromaDB
Developer Tools: Git, GitHub, VSCode, Docker, Linux

Projects
Universal RAG — Multi-Source Grounded Question Answering System
• Built a scalable RAG system indexing PDF, CSV, and Web sources with persistent FAISS vectors and SQLite WAL storage.
• Implemented LangGraph state graph with cyclic query rewriting and retrieval threshold verification, reducing hallucinations by ~40%.
• Designed deterministic citation generation linking directly to physical chunk metadata.

MarketPulse — Multi-Vendor E-Commerce Marketplace
• Engineered a multi-vendor marketplace using MERN stack and LangGraph with atomic inventory concurrency locking.
• Handled 10,000+ mock transactions with sub-50ms query latency across distributed nodes.

Smart Resume Analyzer & Job Recommendation System
• Developed an NLP pipeline using Python, scikit-learn, and NLTK to parse resumes and extract 80+ technical skills.
• Recommended best-fit job roles achieving 88% precision on benchmark candidate profiles.
"""


def test_rubric_weights_sum_to_100():
    """Verify rubric weights sum exactly to 100 points."""
    assert sum(RUBRIC_WEIGHTS.values()) == 100
    assert RUBRIC_WEIGHTS["skills_and_technical_relevance"] == 20
    assert RUBRIC_WEIGHTS["projects_and_demonstrated_impact"] == 20
    assert RUBRIC_WEIGHTS["experience_and_accomplishments"] == 15
    assert RUBRIC_WEIGHTS["education_and_academic_details"] == 10
    assert RUBRIC_WEIGHTS["quantification_and_evidence"] == 10
    assert RUBRIC_WEIGHTS["ats_formatting_and_parseability"] == 10
    assert RUBRIC_WEIGHTS["clarity_grammar_and_conciseness"] == 10
    assert RUBRIC_WEIGHTS["structure_and_consistency"] == 5


def test_rating_band_thresholds():
    """Verify documented rating band thresholds."""
    assert get_rating_band(95.0) == "Exceptional (Top Tier)"
    assert get_rating_band(90.0) == "Exceptional (Top Tier)"
    assert get_rating_band(85.0) == "Strong (Interview Ready)"
    assert get_rating_band(80.0) == "Strong (Interview Ready)"
    assert get_rating_band(75.0) == "Competent (Needs Minor Polish)"
    assert get_rating_band(70.0) == "Competent (Needs Minor Polish)"
    assert get_rating_band(60.0) == "Developing (Needs Significant Improvement)"
    assert get_rating_band(55.0) == "Developing (Needs Significant Improvement)"
    assert get_rating_band(45.0) == "Critical Revision Required"


def test_deterministic_resume_evaluation():
    """Test deterministic evaluation over realistic resume text."""
    result = evaluate_resume_deterministically(
        resume_text=SAMPLE_RESUME_TEXT,
        source_name="Test Resume",
    )

    assert result.overall_score >= 0.0 and result.overall_score <= 100.0
    assert result.rating_band in [
        "Exceptional (Top Tier)",
        "Strong (Interview Ready)",
        "Competent (Needs Minor Polish)",
        "Developing (Needs Significant Improvement)",
        "Critical Revision Required",
    ]
    assert len(result.criterion_scores) == 8

    # Verify all 8 criteria exist with valid bounds
    total_calculated = 0.0
    for key, c in result.criterion_scores.items():
        assert c.score >= 0.0
        assert c.score <= c.weight
        assert len(c.rationale) > 0
        total_calculated += c.score

    assert round(total_calculated, 1) == result.overall_score
    assert len(result.strengths) > 0
    assert len(result.prioritized_improvements) >= 2


def test_bullet_rewrites_contain_star_improvements():
    """Verify weak bullet detection produces constructive rewrites."""
    result = evaluate_resume_deterministically(resume_text=SAMPLE_RESUME_TEXT)
    assert len(result.suggested_rewrites) > 0
    for rewrite in result.suggested_rewrites:
        assert len(rewrite.original_bullet) > 0
        assert len(rewrite.suggested_rewrite) > len(rewrite.original_bullet) or "Architected" in rewrite.suggested_rewrite or "Engineered" in rewrite.suggested_rewrite
        assert len(rewrite.rationale) > 0
        assert len(rewrite.added_impact) > 0


def test_optional_job_description_matching():
    """Verify job description keyword matching and gap analysis."""
    job_desc = """
    We are seeking a Backend AI Engineer with strong proficiency in Python, FastAPI,
    LangChain, Docker, Kubernetes, PostgreSQL, and Rust. Experience building scalable RAG
    applications and optimizing vector databases is required.
    """
    result = evaluate_resume_deterministically(
        resume_text=SAMPLE_RESUME_TEXT,
        job_description=job_desc,
    )

    assert result.job_matching is not None
    jm = result.job_matching
    assert jm.overall_match_percentage >= 0.0 and jm.overall_match_percentage <= 100.0
    assert "python" in jm.matched_required_skills
    assert "fastapi" in jm.matched_required_skills
    assert "rust" in jm.missing_required_skills or "kubernetes" in jm.missing_required_skills
    assert len(jm.recommended_tailoring_changes) > 0
    assert "does not guarantee" in jm.disclaimer


def test_empty_resume_raises_error():
    """Verify empty or whitespace resume raises appropriate exception."""
    with pytest.raises(ValueError):
        evaluate_resume("")

    with pytest.raises(ValueError):
        evaluate_resume("   \n\t  ")


def test_resume_evaluate_api_endpoint():
    """Verify POST /resume/evaluate REST endpoint returns structured response."""
    payload = {
        "raw_text": SAMPLE_RESUME_TEXT,
        "job_description": "Seeking Python engineer with FastAPI and Docker experience.",
    }
    response = client.post("/resume/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "overall_score" in data
    assert "rating_band" in data
    assert "criterion_scores" in data
    assert len(data["criterion_scores"]) == 8
    assert "strengths" in data
    assert "suggested_rewrites" in data
    assert data["job_matching"] is not None
    assert "overall_match_percentage" in data["job_matching"]


def test_rate_my_resume_chat_routing_does_not_fallback():
    """
    Verify querying 'RATE MY RESUME' returns real evaluation instead of fallback refusal:
    'I could not find relevant information in the uploaded sources to answer your question.'
    """
    result = service.query("RATE MY RESUME")
    assert "I could not find relevant information in the uploaded sources to answer your question." not in result["answer"]
    assert "Resume Evaluation" in result["answer"] or "Score" in result["answer"]

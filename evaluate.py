#!/usr/bin/env python3
"""
CLI Evaluation and Benchmarking Suite for Universal RAG.

Usage:
    python evaluate.py
    python evaluate.py --k 4 --compare-baseline
    python evaluate.py --dataset evaluation/benchmark_dataset.json --llm-judge
    python evaluate.py --output evaluation_report.json --markdown-output evaluation_report.md
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.evaluator import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Universal RAG Benchmark & Evaluation Suite")
    parser.add_argument("--k", type=int, default=4, help="Primary Top-K cutoff (default: 4)")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="List of Top-K cutoffs to evaluate (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom evaluation dataset JSON (default: evaluation/benchmark_dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Path to save evaluation output JSON (default: evaluation_report.json)",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default="evaluation_report.md",
        help="Path to save Markdown report (default: evaluation_report.md)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run ablation comparing baseline retrieval against enhanced LangGraph workflow",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable semantic LLM-as-a-judge evaluation (requires configured OpenAI API key)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for LLM judge (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    k_list = args.k_values
    if args.k not in k_list:
        k_list = sorted(list(set(k_list + [args.k])))

    print("==================================================================")
    print("           UNIVERSAL RAG EMPIRICAL BENCHMARK SUITE                ")
    print("==================================================================")
    print(f"Dataset:            {args.dataset or 'evaluation/benchmark_dataset.json'}")
    print(f"Supported K-values: {k_list} (Primary: {args.k})")
    print(f"Baseline Ablation:  {'Enabled' if args.compare_baseline else 'Disabled'}")
    print(f"LLM-as-a-Judge:     {'Enabled' if args.llm_judge else 'Disabled (Offline/Heuristic mode)'}")
    print("------------------------------------------------------------------")
    print("Running evaluation across benchmark samples...\n")

    evaluator = RAGEvaluator(
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
        judge_model=args.judge_model,
    )
    results = evaluator.run_evaluation(
        k_values=k_list,
        compare_baseline=args.compare_baseline,
    )

    summary = results["summary"]
    run_id = summary["run_id"]

    print("==================================================================")
    print("                    EVALUATION RESULTS                            ")
    print("==================================================================")
    print(f"Total Benchmark Queries:     {summary['total_samples']}")
    print(f"Answerable Queries:          {summary['answerable_samples']}")
    print(f"Unanswerable Queries:        {summary['unanswerable_samples']}")
    print("------------------------------------------------------------------")
    print("Retrieval Metrics:")
    for k in k_list:
        rec = summary.get(f"mean_recall@{k}", 0.0)
        prec = summary.get(f"mean_precision@{k}", 0.0)
        ndcg = summary.get(f"mean_ndcg@{k}", 0.0)
        print(f"  Cutoff K={k:2d}:  Recall@{k}: {rec:.4f} | Precision@{k}: {prec:.4f} | nDCG@{k}: {ndcg:.4f}")
    print(f"Mean Reciprocal Rank (MRR):  {summary['mean_mrr']:.4f}")
    print("------------------------------------------------------------------")
    print("Citation & Abstention Quality:")
    print(f"Citation Precision:          {summary['mean_citation_precision']:.4f}")
    print(f"Citation Coverage:           {summary['mean_citation_coverage']:.4f}")
    print(f"Abstention Accuracy:         {summary['abstention_accuracy']:.4f}")
    print("------------------------------------------------------------------")
    print("Operational Latency Profile:")
    print(f"Mean Retrieval Latency:      {summary['mean_retrieval_latency_ms']:.2f} ms")
    print(f"Mean End-to-End Latency:     {summary['mean_e2e_latency_ms']:.2f} ms")
    print(f"p50 Latency (Median):        {summary['latency_p50_ms']:.2f} ms")
    print(f"p95 Latency:                 {summary['latency_p95_ms']:.2f} ms")

    if "ablation_comparison" in results:
        ab = results["ablation_comparison"]
        print("------------------------------------------------------------------")
        print("Ablation Study (Baseline vs Enhanced LangGraph Workflow):")
        print(f"  Baseline Recall@{ab['k']}:  {ab['baseline_mean_recall']:.4f}")
        print(f"  Enhanced Recall@{ab['k']}:  {ab['enhanced_mean_recall']:.4f}")
        print(f"  Improvement:            +{ab['recall_improvement']:.4f}")

    if "judge_faithfulness" in summary:
        print("------------------------------------------------------------------")
        print(f"LLM-as-a-Judge Evaluation ({summary.get('judge_model')}):")
        print(f"  Claim Faithfulness:     {summary['judge_faithfulness']:.4f}")
        print(f"  Answer Correctness:     {summary['judge_correctness']:.4f}")
        print(f"  Answer Relevance:       {summary['judge_relevance']:.4f}")
    print("==================================================================")

    # 1. Save machine-readable JSON report
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Save historical versioned report in evaluation/reports/
    reports_dir = Path("evaluation") / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / f"{run_id}_report.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 3. Save Markdown report
    md_content = evaluator.generate_markdown_report(results)
    md_output_path = Path(args.markdown_output)
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    history_md = reports_dir / f"{run_id}_report.md"
    with open(history_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[+] Full JSON report saved to:     {output_path.resolve()}")
    print(f"[+] Full Markdown report saved to: {md_output_path.resolve()}")
    print(f"[+] Versioned archive saved to:   {history_path.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()

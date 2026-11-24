# Runs the end-to-end Plain RAG evaluation, computes Execution Accuracy (ExAcc) (handling both exact scale match and % vs decimal scale differences like 0.125 vs 12.5), and saves a detailed JSONL log for your Chapter 6 error analysis.   
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from src.baseline.generator import BaselineGenerator
from src.baseline.retriever import BaselineRetriever
from src.ingestion.finqa_loader import load_finqa_split


def is_numerically_correct(
    pred: Optional[float],
    gold: Optional[float],
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-2,
) -> bool:
    """
    Checks Execution Accuracy (ExAcc) within tolerance, accounting for percentage scale
    differences (e.g., 0.152 vs 15.2) common in FinQA ground truth.
    """
    if pred is None or gold is None:
        return False

    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= max(abs_tol, rel_tol * abs(b))

    if _close(pred, gold):
        return True
    # Check percentage scaling (100x difference)
    if _close(pred / 100.0, gold) or _close(pred * 100.0, gold):
        return True
    return False


def run_baseline_eval(
    split_path: str,
    limit: int,
    model_name: str,
    top_k: int,
    output_path: str,
) -> None:
    records = load_finqa_split(split_path, limit=limit)
    retriever = BaselineRetriever(top_k=top_k)
    generator = BaselineGenerator(model_name=model_name)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_evaluable = 0
    correct_count = 0
    parse_failures = 0
    total_latency = 0.0
    total_tokens = 0

    with out_file.open("w", encoding="utf-8") as f:
        for rec in tqdm(records, desc=f"Evaluating Plain RAG ({model_name})"):
            if rec.gold_answer is None:
                continue

            total_evaluable += 1
            chunks = retriever.retrieve(rec.question, doc_id=rec.doc_id)
            gen_res = generator.generate(rec.question, chunks)

            pred = gen_res["predicted_answer"]
            if pred is None:
                parse_failures += 1

            is_correct = is_numerically_correct(pred, rec.gold_answer)
            if is_correct:
                correct_count += 1

            total_latency += gen_res["latency_sec"]
            total_tokens += gen_res["prompt_tokens"] + gen_res["completion_tokens"]

            log_entry = {
                "doc_id": rec.doc_id,
                "question": rec.question,
                "gold_answer": rec.gold_answer,
                "gold_program": rec.program,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "latency_sec": gen_res["latency_sec"],
                "prompt_tokens": gen_res["prompt_tokens"],
                "completion_tokens": gen_res["completion_tokens"],
                "raw_output": gen_res["raw_output"],
            }
            f.write(json.dumps(log_entry) + "\n")

    ex_acc = (correct_count / total_evaluable * 100.0) if total_evaluable else 0.0
    avg_latency = (total_latency / total_evaluable) if total_evaluable else 0.0
    avg_tokens = (total_tokens / total_evaluable) if total_evaluable else 0.0

    summary = {
        "system": "Plain RAG Baseline",
        "model": model_name,
        "top_k": top_k,
        "total_evaluated": total_evaluable,
        "correct": correct_count,
        "execution_accuracy_pct": round(ex_acc, 2),
        "parse_failures": parse_failures,
        "avg_latency_sec": round(avg_latency, 3),
        "avg_tokens_per_query": round(avg_tokens, 1),
    }

    print("\n" + "=" * 55)
    print("PLAIN RAG BASELINE EVALUATION SUMMARY")
    print("=" * 55)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Detailed logs saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Plain RAG Baseline on FinQA.")
    parser.add_argument("--split", type=str, default="data/finqa/dev.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/baseline_dev_200.jsonl")
    args = parser.parse_args()

    run_baseline_eval(
        split_path=args.split,
        limit=args.limit,
        model_name=args.model,
        top_k=args.top_k,
        output_path=args.output,
    )
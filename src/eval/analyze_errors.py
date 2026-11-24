from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def classify_baseline_failure(entry: Dict[str, Any]) -> str:
    """
    Categorizes a failed Plain RAG baseline prediction into distinct RQ2 failure modes:
    1. Parse / Token Truncation Error: Model failed to emit a parseable FINAL_ANSWER.
    2. Scale / Sign Mismatch: Magnitude matches within 10x/1000x (e.g., millions vs raw units) or inverted sign.
    3. Multi-Step Arithmetic / Program Error: Ground-truth program requires >= 2 operations and LLM miscalculated.
    4. Single-Step Lookup / Arithmetic Hallucination: Single operation or direct cell lookup failed due to chunk fragmentation or math error.
    """
    pred = entry.get("predicted_answer")
    gold = entry.get("gold_answer")
    program = entry.get("gold_program") or ""
    raw_output = entry.get("raw_output") or ""

    # 1. Output extraction / truncation failure
    if pred is None or "FINAL_ANSWER:" not in raw_output:
        return "Parse / Output Truncation Error"

    if gold is None:
        return "Other"

    # 2. Sign inversion (e.g., -45.2 vs 45.2)
    if gold != 0 and abs(pred + gold) <= max(1e-2, 1e-2 * abs(gold)):
        return "Sign Inversion Error"

    # 3. Unit / Magnitude scaling error (e.g., millions vs thousands: 1000x or 1000000x)
    if gold != 0:
        ratio = abs(pred / gold)
        for scale in (10.0, 1000.0, 1_000_000.0):
            if abs(ratio - scale) / scale < 0.02 or abs(ratio - (1.0 / scale)) / (1.0 / scale) < 0.02:
                return "Unit / Magnitude Scaling Error"

    # 4. Multi-step vs Single-step reasoning error based on FinQA gold program operations
    # FinQA programs format steps like: "subtract(5829, 5735), divide(#0, 5735)"
    step_count = program.count("(")
    if step_count >= 2:
        return "Multi-Step Arithmetic / Reasoning Error"

    return "Single-Step Lookup / Hallucination Error"


def analyze_results(input_path: str, output_report_path: str) -> None:
    in_file = Path(input_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Baseline log file not found at: {in_file}")

    records: List[Dict[str, Any]] = []
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    correct_records = [r for r in records if r.get("is_correct")]
    failed_records = [r for r in records if not r.get("is_correct")]

    error_counts: Counter[str] = Counter()
    categorized_failures: List[Dict[str, Any]] = []

    for entry in failed_records:
        category = classify_baseline_failure(entry)
        error_counts[category] += 1
        categorized_failures.append(
            {
                "doc_id": entry.get("doc_id"),
                "failure_category": category,
                "question": entry.get("question"),
                "gold_program": entry.get("gold_program"),
                "gold_answer": entry.get("gold_answer"),
                "predicted_answer": entry.get("predicted_answer"),
                "latency_sec": entry.get("latency_sec"),
            }
        )

    # Latency comparison between correct vs failed queries
    avg_lat_correct = (
        sum(r.get("latency_sec", 0.0) for r in correct_records) / len(correct_records)
        if correct_records
        else 0.0
    )
    avg_lat_failed = (
        sum(r.get("latency_sec", 0.0) for r in failed_records) / len(failed_records)
        if failed_records
        else 0.0
    )

    report = {
        "input_file": str(in_file),
        "total_evaluated": total,
        "correct_count": len(correct_records),
        "failed_count": len(failed_records),
        "execution_accuracy_pct": round((len(correct_records) / total * 100.0) if total else 0.0, 2),
        "avg_latency_correct_sec": round(avg_lat_correct, 3),
        "avg_latency_failed_sec": round(avg_lat_failed, 3),
        "failure_taxonomy": {
            cat: {
                "count": count,
                "pct_of_failures": round((count / len(failed_records) * 100.0) if failed_records else 0.0, 2),
                "pct_of_total": round((count / total * 100.0) if total else 0.0, 2),
            }
            for cat, count in error_counts.most_common()
        },
        "sample_failures": categorized_failures[:25],
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("PLAIN RAG BASELINE — RQ2 FAILURE MODE ANALYSIS")
    print("=" * 60)
    print(f"Total Evaluated : {total}")
    print(f"Correct         : {len(correct_records)} ({report['execution_accuracy_pct']}%)")
    print(f"Failed          : {len(failed_records)}")
    print(f"Avg Latency (Correct vs Failed): {avg_lat_correct:.2f}s vs {avg_lat_failed:.2f}s\n")
    print("Failure Breakdown:")
    for cat, stats in report["failure_taxonomy"].items():
        print(
            f"  - {cat:<42} : {stats['count']:>3} "
            f"({stats['pct_of_failures']:>5.1f}% of errors | {stats['pct_of_total']:>4.1f}% of total)"
        )
    print(f"\nFull taxonomy report saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Plain RAG baseline failure modes.")
    parser.add_argument("--input", type=str, default="results/baseline_dev_200.jsonl")
    parser.add_argument("--output", type=str, default="results/baseline_error_taxonomy.json")
    args = parser.parse_args()

    analyze_results(input_path=args.input, output_report_path=args.output)
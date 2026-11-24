# Calls your local Ollama model (qwen2.5:7b or llama3.1:8b), enforces a strict structured tag (FINAL_ANSWER: <number>) so numeric extraction is reliable, and records latency + token counts for RQ3 (accuracy/latency/cost trade-offs).
from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Optional
import ollama


SYSTEM_PROMPT = (
    "You are a financial analyst answering quantitative questions strictly from the provided context. "
    "Show concise step-by-step arithmetic, then end your response with a single line formatted exactly as:\n"
    "FINAL_ANSWER: <numeric_value>\n"
    "Do not include currency symbols, commas, or units after FINAL_ANSWER:. "
    "If the answer is a percentage, output the numeric value as asked (e.g., 12.5 for 12.5%)."
)


def extract_final_number(response_text: str) -> Optional[float]:
    """Parses the numeric value following FINAL_ANSWER:, falling back to the last number in text."""
    match = re.search(r"FINAL_ANSWER:\s*([+-]?\d+(?:\.\d+)?)", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Fallback: scan for standalone numbers in the final line
    lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
    if lines:
        cleaned = lines[-1].replace(",", "").replace("$", "").replace("%", "")
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", cleaned)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                return None
    return None


class BaselineGenerator:
    """Plain RAG generator using local Ollama inference."""

    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_blocks = [
            f"[Chunk {i + 1}]\n{chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
        ]
        context_str = "\n\n---\n\n".join(context_blocks)

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        start_time = time.perf_counter()
        # response = ollama.chat(
        #     model=self.model_name,
        #     messages=[
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     options={"temperature": self.temperature},
        # )
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_ctx": 2048,
                "num_predict": 256,
            },
        )
        latency_sec = time.perf_counter() - start_time

        raw_output = response["message"]["content"]
        predicted_num = extract_final_number(raw_output)

        return {
            "raw_output": raw_output,
            "predicted_answer": predicted_num,
            "latency_sec": round(latency_sec, 3),
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
        }
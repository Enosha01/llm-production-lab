"""
Local benchmark: latency and throughput of the FastAPI inference server.
Compares short vs medium vs long prompts.
Outputs results to results/local_bench.csv
"""

import time
import csv
import httpx
from pathlib import Path

API_URL = "http://localhost:8001/generate"
OUTPUT = Path("results/local_bench.csv")

PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain what a transformer neural network is in three sentences.",
    "long": (
        "Explain the history of artificial intelligence in detail. "
        "Start from the 1950s Dartmouth conference, cover symbolic AI, "
        "the AI winters, expert systems, machine learning, deep learning, "
        "and end with the current era of large language models. "
        "Be thorough and include specific dates, researchers, and milestones."
    )
}

MAX_TOKENS = 100
RUNS_PER_PROMPT = 3


def run_single(prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    start = time.time()
    response = httpx.post(
        API_URL,
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=120.0
    )
    total_ms = (time.time() - start) * 1000
    data = response.json()

    return {
        "latency_ms": round(total_ms, 2),
        "server_latency_ms": data.get("latency_ms", 0),
        "output_chars": len(data.get("text", "")),
        "text": data.get("text", "")[:80]
    }


def run_benchmark():
    OUTPUT.parent.mkdir(exist_ok=True)
    results = []

    print("Starting benchmark...")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Runs per prompt: {RUNS_PER_PROMPT}")
    print("-" * 60)

    for label, prompt in PROMPTS.items():
        print(f"\nPrompt type: {label} ({len(prompt)} chars)")
        for i in range(RUNS_PER_PROMPT):
            try:
                r = run_single(prompt)
                r["prompt_type"] = label
                r["prompt_chars"] = len(prompt)
                r["run"] = i + 1
                results.append(r)
                print(
                    f"  Run {i+1}: {r['latency_ms']} ms | "
                    f"{r['output_chars']} chars | {r['text'][:40]}..."
                )
            except Exception as e:
                print(f"  Run {i+1}: ERROR — {e}")

    # Write CSV
    if results:
        with open(OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved to {OUTPUT}")
    else:
        print("\n❌ No results to save")


if __name__ == "__main__":
    run_benchmark()
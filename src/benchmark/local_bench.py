"""
Local llama.cpp benchmark wrapper for M1 Metal.

Runs llama-bench across thread counts and saves results.
"""

import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime


MODEL = "models/phi3-mini-q4.gguf"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_bench(model: str, threads: str, out_file: str):
    """Run llama-bench and save output to file."""
    cmd = [
        "llama-bench",
        "-m", model,
        "-t", threads,
        "-n", "128",
        "-p", "512",
        "-r", "3",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    # Save output
    with open(out_file, "w") as f:
        f.write(f"# llama-bench run\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# Model: {model}\n")
        f.write(f"# Command: {' '.join(cmd)}\n\n")
        f.write(result.stdout)

    print(result.stdout)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        print(f"Model not found: {MODEL}")
        print("Download Phi-3-mini GGUF first.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = str(RESULTS_DIR / f"llama_bench_m1_{timestamp}.txt")

    run_bench(MODEL, "2,4,6,8", out_file)

"""
LLM Inference Server
FastAPI wrapper that will connect to vLLM (cloud) or Ollama (local).
"""

import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

# Configuration
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:11434/v1")  # Ollama default
MODEL_NAME = os.getenv("MODEL_NAME", "phi3:mini")

app = FastAPI(
    title="LLM Inference Server",
    description="Production LLM serving with vLLM/Ollama",
    version="0.1.0"
)

# OpenAI-compatible client
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")


# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    model: str
    latency_ms: float


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "service": "LLM Inference Server",
        "model": MODEL_NAME,
        "backend": VLLM_URL,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text from a prompt (non-streaming)."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    start = time.time()
    try:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        latency_ms = (time.time() - start) * 1000

        return GenerateResponse(
            text=response.choices[0].text,
            model=MODEL_NAME,
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version")
def version():
    return {"version": "0.1.0", "day": 2, "status": "skeleton"}
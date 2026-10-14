"""
LLM Inference Server — FastAPI wrapper.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="LLM Inference Server",
    description="Production LLM serving with vLLM",
    version="0.1.0",
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int


@app.get("/")
def root():
    return {
        "service": "llm-inference-server",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    return GenerateResponse(
        text=f"[PLACEHOLDER] You asked: {req.prompt}",
        model="placeholder",
        tokens_generated=0,
    )

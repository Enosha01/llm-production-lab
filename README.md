# LLM Production Lab

An end-to-end LLM serving system with vLLM, FastAPI, Redis, and benchmarks.

## Status

- [x] Day 1: Setup + Ollama (Phi-3-mini) running locally
- [x] Day 2: FastAPI server + local benchmarks
- [ ] Layer 1: vLLM inference server on RunPod
- [ ] Layer 2: RAG pipeline
- [ ] Layer 3: Optimization benchmarks

## Architecture

**Local (M1 Mac Air 8GB):**
- Ollama (Phi-3-mini) for dev
- FastAPI server
- Redis cache

**Cloud (RunPod GPU):**
- vLLM serving Llama-3.2-3B
- Quantization benchmarks
- Load testing

## Stack

- Inference: vLLM, Ollama, llama.cpp
- API: FastAPI, Uvicorn
- Cache: Redis
- Monitoring: Prometheus, Grafana
- Deploy: Docker, HuggingFace Spaces
- Testing: pytest, Locust

## Local Benchmarks (Day 2)

Environment: Mac M1 Air 8GB, Ollama, Phi-3-mini

| Prompt Type | Avg Latency | Notes |
|---|---|---|
| Short (12 chars) | 554 ms | Cache hit: 114 ms |
| Medium (64 chars) | 3779 ms | ~500 chars output |
| Long (308 chars) | 4074 ms | Minimal increase vs medium |

**Key insight:** Prompt length has minimal impact on latency. Generation dominates. Cache hits reduce latency by 10x.

## Roadmap

- Days 1-14: Layer 1 (vLLM server)
- Days 15-30: Layer 2 (RAG + eval)
- Days 31-50: Layer 3 (Benchmarks)
- Days 51-56: Polish + launch

## Author

Enosha — CSE (AI/ML), Presidency University, Bangalore
GitHub: [@Enosha01](https://github.com/Enosha01)
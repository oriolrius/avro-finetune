# 🚀 Model Deployment Guide

This guide covers deploying your fine-tuned models to production using vLLM and Ollama with Docker.

## Table of Contents
- [Overview](#overview)
- [Export Pipeline](#export-pipeline)
- [vLLM Deployment](#vllm-deployment)
- [Ollama Deployment](#ollama-deployment)
- [Performance Comparison](#performance-comparison)
- [Troubleshooting](#troubleshooting)

## Overview

After fine-tuning with QLoRA, you have several deployment options:
- **vLLM**: High-performance inference server optimized for throughput
- **Ollama**: User-friendly local deployment with model management
- **Hugging Face**: Direct integration with transformers library

### Key Concepts

1. **LoRA Adapters**: Small weight files (~3MB) from fine-tuning
2. **Merged Model**: Base model + adapters combined (~7GB)
3. **Quantization**: Optional size reduction (GGUF for Ollama)
4. **Deployment Format**: Platform-specific model packaging

## Export Pipeline

### Automatic Export (During Training)

Add these to your `.env` file:
```bash
# Export configuration
EXPORT_VLLM=true        # Export for vLLM after training
EXPORT_OLLAMA=true      # Export for Ollama after training
AUTO_MERGE=true         # Automatically merge and export
```

Then train normally:
```bash
uv run python train_configurable.py
```

### Manual Export (After Training)

#### List Available Experiments
```bash
uv run python merge_and_export.py --list
```

#### Export Latest Experiment
```bash
# For vLLM
uv run python merge_and_export.py --latest --format vllm

# For Ollama
uv run python merge_and_export.py --latest --format ollama

# For Hugging Face
uv run python merge_and_export.py --latest --format huggingface
```

#### Export Specific Experiment
```bash
uv run python merge_and_export.py \
  --experiment phi3mini4k-minimal-r32-a64-e20-20240914-143022 \
  --format vllm
```

## vLLM Deployment

### What is vLLM?

vLLM is a high-throughput, memory-efficient inference engine for LLMs. It features:
- PagedAttention for efficient memory management
- Continuous batching for high throughput
- OpenAI-compatible API server
- Tensor parallelism support

### Export for vLLM

```bash
# Export model
uv run python merge_and_export.py \
  --latest \
  --format vllm \
  --dtype float16

# This creates:
# exports/{experiment}-vllm-{timestamp}/
# ├── config.json
# ├── model.safetensors
# ├── tokenizer.json
# ├── vllm_config.json
# ├── docker-compose.vllm.yml
# └── test_vllm.sh
```

### Deploy with Docker

```bash
# Navigate to export directory
cd exports/phi3mini4k-minimal-r32-a64-e20-vllm-*

# Start vLLM server
docker-compose -f docker-compose.vllm.yml up -d

# Wait for server to start (check logs)
docker-compose -f docker-compose.vllm.yml logs -f

# Test the deployment
./test_vllm.sh
```

### Use vLLM API

#### Python Client
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "/model",
        "prompt": "Create an AVRO schema for a user",
        "max_tokens": 200,
        "temperature": 0.7
    }
)

print(json.loads(response.text)["choices"][0]["text"])
```

#### Curl Example
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "prompt": "Create an AVRO schema for a product",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

#### OpenAI SDK Compatible
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't need a real key
)

response = client.completions.create(
    model="/model",
    prompt="Create an AVRO schema for an order",
    max_tokens=200
)

print(response.choices[0].text)
```

### vLLM Performance Tuning

```yaml
# docker-compose.vllm.yml adjustments
services:
  vllm:
    command: >
      --model /model
      --dtype float16
      --max-model-len 4096
      --max-num-seqs 256        # Max concurrent requests
      --max-num-batched-tokens 8192
      --gpu-memory-utilization 0.9
      --trust-remote-code
```

## Ollama Deployment

### What is Ollama?

Ollama provides a simple way to run LLMs locally with:
- Easy model management
- Built-in quantization support
- Simple CLI and API
- Model versioning
- GGUF format support

### Export for Ollama

```bash
# Export model
uv run python merge_and_export.py \
  --latest \
  --format ollama

# This creates:
# exports/{experiment}-ollama-{timestamp}/
# ├── pytorch_model.bin
# ├── config.json
# ├── tokenizer.json
# ├── Modelfile
# ├── OLLAMA_INSTRUCTIONS.md
# ├── docker-compose.ollama.yml
# └── test_ollama.sh
```

### Convert to GGUF (Required for Ollama)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert to GGUF
cd ../exports/phi3mini4k-minimal-r32-a64-e20-ollama-*
python ../llama.cpp/convert.py . \
  --outtype f16 \
  --outfile model.gguf

# Optional: Quantize for smaller size
../llama.cpp/quantize model.gguf model-q4_k_m.gguf q4_k_m
```

### Deploy with Docker

```bash
# Start Ollama server
docker-compose -f docker-compose.ollama.yml up -d

# Create model in Ollama
docker exec -it ollama ollama create my-phi3 -f /models/Modelfile

# Test the model
./test_ollama.sh
```

### Use Ollama

#### CLI Usage
```bash
# Interactive chat
docker exec -it ollama ollama run my-phi3

# Single prompt
docker exec -it ollama ollama run my-phi3 "Create an AVRO schema for a user"
```

#### API Usage
```python
import requests
import json

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "my-phi3",
        "prompt": "Create an AVRO schema for a product",
        "stream": False
    }
)

result = json.loads(response.text)
print(result["response"])
```

#### Streaming API
```python
import requests
import json

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "my-phi3",
        "prompt": "Create an AVRO schema for an order",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(chunk["response"], end="", flush=True)
```

## Performance Comparison

### Deployment Options Comparison

| Feature | vLLM | Ollama | Direct HF |
|---------|------|---------|-----------|
| **Throughput** | High (batched) | Medium | Low |
| **Latency** | Low | Medium | High |
| **Memory Usage** | Optimized | Good (quantized) | High |
| **Setup Complexity** | Medium | Low | Low |
| **Quantization** | Post-export | Built-in | Manual |
| **API** | OpenAI-compatible | REST | Python only |
| **Best For** | Production APIs | Local development | Research |

### Benchmarking Script

```python
#!/usr/bin/env python3
"""benchmark.py - Compare deployment performance"""

import time
import requests
import statistics

def benchmark_vllm(prompt, n=10):
    times = []
    for _ in range(n):
        start = time.time()
        requests.post(
            "http://localhost:8000/v1/completions",
            json={"model": "/model", "prompt": prompt, "max_tokens": 200}
        )
        times.append(time.time() - start)
    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times)
    }

def benchmark_ollama(prompt, n=10):
    times = []
    for _ in range(n):
        start = time.time()
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "my-phi3", "prompt": prompt, "stream": False}
        )
        times.append(time.time() - start)
    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times)
    }

# Run benchmarks
prompt = "Create an AVRO schema for a user"
print("Benchmarking vLLM...")
vllm_stats = benchmark_vllm(prompt)
print(f"vLLM: {vllm_stats}")

print("Benchmarking Ollama...")
ollama_stats = benchmark_ollama(prompt)
print(f"Ollama: {ollama_stats}")
```

## Production Deployment

### vLLM in Production

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./model:/model:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # Multi-GPU
    command: >
      --model /model
      --dtype float16
      --tensor-parallel-size 2    # Use 2 GPUs
      --max-model-len 4096
      --max-num-seqs 256
      --gpu-memory-utilization 0.95
      --trust-remote-code
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - vllm
```

### Monitoring

```yaml
# Add to docker-compose
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce batch size for vLLM
--max-num-seqs 64
--gpu-memory-utilization 0.8

# Use quantized model for Ollama
./quantize model.gguf model-q4_k_m.gguf q4_k_m
```

#### Slow Inference
```bash
# Enable Flash Attention for vLLM
--use-flash-attn

# Use smaller max length
--max-model-len 2048
```

#### Docker GPU Access
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Health Checks

```bash
# vLLM health check
curl http://localhost:8000/health

# Ollama health check
curl http://localhost:11434/api/tags

# Model info
curl http://localhost:8000/v1/models
```

## Best Practices

1. **Model Selection**:
   - Use vLLM for high-throughput production APIs
   - Use Ollama for local development and testing
   - Consider quantization for edge deployment

2. **Resource Management**:
   - Monitor GPU memory usage
   - Set appropriate batch sizes
   - Use model sharding for large models

3. **Security**:
   - Use API keys in production
   - Implement rate limiting
   - Run behind reverse proxy (nginx)

4. **Optimization**:
   - Profile your specific workload
   - Adjust parameters based on usage patterns
   - Consider model quantization trade-offs

## Next Steps

- [Kubernetes Deployment Guide](./KUBERNETES.md)
- [Model Optimization Guide](./OPTIMIZATION.md)
- [API Documentation](./API.md)
- [Monitoring Setup](./MONITORING.md)
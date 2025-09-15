# GitHub Actions Workflows Guide

This directory contains GitHub Actions workflows for training, exporting, and publishing Phi-3 models fine-tuned for AVRO schema generation.

## Active Workflows (5 total)

### 🚀 Production Workflows

#### 1. `train-and-export.yml` - Main Training Pipeline
**Purpose**: Complete training pipeline with integrated export functionality

**Triggers**:
- Manual dispatch via GitHub UI
- Git tags: `train-*`, `exp-*`

**Features**:
- Configurable training parameters (epochs, batch size, learning rate, LoRA settings)
- Automatic export to vLLM and Ollama formats after training
- Calls `export-complete.yml` as a reusable workflow
- Comprehensive artifact generation

**Usage**:
```bash
# Trigger via GitHub CLI
gh workflow run train-and-export.yml \
  -f num_epochs=20 \
  -f batch_size=2 \
  -f export_formats=all
```

---

#### 2. `export-complete.yml` - Export Pipeline (Reusable)
**Purpose**: Standalone export workflow for converting adapters to deployment formats

**Triggers**:
- Manual dispatch via GitHub UI
- Called by other workflows (reusable)

**Features**:
- Exports LoRA adapters to vLLM format
- Exports to Ollama GGUF format with configurable quantization
- Creates deployment packages with Docker configurations
- Generates comprehensive deployment instructions

**Outputs**:
- vLLM model package (`.tar.gz`)
- Ollama GGUF model package (`.tar.gz`)
- Deployment instructions

**Usage**:
```bash
# Export latest adapter to all formats
gh workflow run export-complete.yml \
  -f export_formats=all \
  -f ollama_quantization=q4_k_m
```

---

#### 3. `publish-huggingface.yml` - Model Publishing
**Purpose**: Publishes trained models to Hugging Face Hub

**Triggers**: Manual dispatch only

**Supported Model Types**:
- **vLLM**: Full models optimized for vLLM deployment
- **Ollama**: GGUF quantized models
- **Adapter**: LoRA adapters

**Features**:
- Automatic metadata generation (prevents YAML warnings)
- Comprehensive README creation with usage examples
- Support for private repositories
- Auto-generates repository names with timestamps

**Usage**:
```bash
# Publish latest vLLM model
gh workflow run publish-huggingface.yml \
  -f model_type=vllm

# Publish Ollama model with custom name
gh workflow run publish-huggingface.yml \
  -f model_type=ollama \
  -f repo_name=phi3-avro-gguf-v2
```

---

### 🧪 Development/Testing Workflows

#### 4. `test-setup.yml` - Environment Testing
**Purpose**: Tests CI/CD environment setup and configuration

**Features**:
- Validates Docker installation
- Tests Python/uv setup
- Verifies repository structure
- Creates mock artifacts for testing
- Useful for debugging CI issues

**When to use**: When setting up new runners or debugging environment issues

---

#### 5. `test-deployment.yml` - Deployment Testing
**Purpose**: Tests deployment configurations without real models

**Features**:
- Mock vLLM server testing
- Mock Ollama server testing
- Deployment benchmarking
- Docker compose validation

**When to use**: Before deploying to production or testing new deployment configurations

---

## Workflow Architecture

```
train-and-export.yml
        ↓ (calls)
export-complete.yml
        ↓ (produces)
   Model Artifacts
        ↓ (consumed by)
publish-huggingface.yml
        ↓ (publishes to)
    Hugging Face Hub
```

## Key Features

### Self-Hosted Runners
Workflows use self-hosted runners (`[self-hosted, Linux, X64]`) for:
- Memory-intensive operations (model merging, quantization)
- GPU-accelerated tasks
- Large file handling (7GB+ GGUF files)

### Artifact Management
- Artifacts are retained for 7 days (production) or 1 day (testing)
- Named with timestamps for versioning
- Compressed as `.tar.gz` for efficient storage

### Environment Variables
Required secrets:
- `HF_TOKEN`: Hugging Face API token for model access and publishing

## Common Operations

### Full Pipeline: Train → Export → Publish
```bash
# 1. Train and export
gh workflow run train-and-export.yml \
  -f num_epochs=20 \
  -f export_formats=all

# 2. Wait for completion, then publish
gh workflow run publish-huggingface.yml \
  -f model_type=ollama \
  -f repo_name=my-phi3-model
```

### Export Existing Adapter
```bash
gh workflow run export-complete.yml \
  -f adapter_path=phi3mini4k-minimal-r32-a64-e20-20250914-132416 \
  -f export_formats=ollama \
  -f ollama_quantization=q5_k_m
```

### Test New Deployment Configuration
```bash
gh workflow run test-deployment.yml
```

## Maintenance Notes

- Workflows use `uv` for fast Python dependency management
- Docker is required for Ollama GGUF conversion
- All workflows include comprehensive error handling and logging
- Summaries are generated in GitHub UI for easy monitoring

## Removed Workflows

- `finetune-and-evaluate.yml` - Obsolete, replaced by `train-and-export.yml`
  - Used older training script (`train.py`)
  - Lacked export integration
  - Simple artifact handling without deployment support
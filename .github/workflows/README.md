# GitHub Actions Workflows

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
- Comprehensive artifact generation and pipeline summary

**Inputs**:
- `dataset_path`: Training dataset (default: 'dataset_minimal.jsonl')
- `num_epochs`: Training epochs (default: 20)
- `batch_size`: Batch size (default: 2)
- `learning_rate`: Learning rate (default: '5e-5')
- `lora_rank`: LoRA rank parameter (default: 32)
- `lora_alpha`: LoRA alpha parameter (default: 64)
- `export_formats`: 'all', 'vllm', 'ollama', or 'none' (default: 'all')
- `ollama_quantization`: GGUF quantization format (default: 'q4_k_m')

**Usage**:
```bash
# Trigger via GitHub CLI
gh workflow run train-and-export.yml \
  -f num_epochs=20 \
  -f batch_size=2 \
  -f export_formats=all
```

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
- No cross-workflow artifact issues

**Inputs**:
- `adapter_path`: Adapter directory name (default: 'phi3mini4k-minimal-r32-a64-e20-20250914-132416')
- `export_formats`: Choose 'all', 'vllm', or 'ollama' (default: 'all')
- `ollama_quantization`: Quantization format - q4_k_m, q5_k_m, q8_0, f16 (default: 'q4_k_m')

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

### 🧪 Development/Testing Workflows

#### 4. `test-deployment.yml` - Deployment Testing
**Purpose**: Test deployed vLLM and Ollama models
**Trigger**: Manual only (workflow_dispatch)
**Features**:
- Tests vLLM OpenAI-compatible API
- Tests Ollama API endpoints
- Runs inference benchmarks
- Validates deployment health

**Inputs**:
- `deployment_type`: 'vllm', 'ollama', or 'both' (default: 'both')
- `model_name`: Model name for Ollama (default: 'phi3-avro')
- `test_prompt`: Test prompt for inference (default: 'What is the capital of France?')

**Usage**:
```bash
gh workflow run test-deployment.yml \
  -f deployment_type="both" \
  -f model_name="phi3-avro" \
  -f test_prompt="Explain AI in simple terms"
```

#### 5. `test-setup.yml` - Environment Testing
**Purpose**: Tests CI/CD environment setup and configuration

**Features**:
- Validates Docker installation
- Tests Python/uv setup
- Verifies repository structure
- Creates mock artifacts for testing
- Useful for debugging CI issues

**When to use**: When setting up new runners or debugging environment issues

**Usage**:
```bash
gh workflow run test-setup.yml
```

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

### Key Features

#### Self-Hosted Runners
Workflows use self-hosted runners (`[self-hosted, Linux, X64]`) for:
- Memory-intensive operations (model merging, quantization)
- GPU-accelerated tasks
- Large file handling (7GB+ GGUF files)

#### Artifact Management
- Artifacts are retained for 7 days (production) or 1 day (testing)
- Named with timestamps for versioning
- Compressed as `.tar.gz` for efficient storage

#### Environment Variables
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

## Quick Start Guide

### Option 1: Complete Training + Export
```bash
# Train a model and export it
gh workflow run train-and-export.yml \
  -f num_epochs=30 \
  -f export_formats="all"

# Or trigger with a tag
git tag train-$(date +%Y%m%d)
git push origin train-$(date +%Y%m%d)
```

### Option 2: Export Existing Adapter
```bash
# Export an existing adapter
gh workflow run export-complete.yml \
  -f adapter_path="your-adapter-name" \
  -f export_formats="all"
```

### Option 3: Test Deployments
```bash
# Test your deployments
gh workflow run test-deployment.yml \
  -f deployment_type="both"
```

## Deployment Instructions

### From Export Pipeline

1. Run the export workflow:
```bash
gh workflow run export-complete.yml -f export_formats="all"
```

2. Download artifacts from the workflow run page

3. Deploy vLLM:
```bash
tar -xzf vllm-*.tar.gz
docker compose up -d
```

4. Deploy Ollama:
```bash
tar -xzf ollama-*.tar.gz
./setup.sh
```

## Environment Requirements

### GitHub-hosted Runners
- Ubuntu latest
- Docker and Docker Compose pre-installed
- Python 3.10+
- No GPU required for export workflows

### Self-hosted Runners (for training)
- Ubuntu 20.04+ or WSL2
- CUDA 12.8+
- Python 3.10+
- At least 16GB RAM
- At least 50GB free disk space
- GPU with 8GB+ VRAM

## Secrets Required
- `HF_TOKEN`: Hugging Face API token (for model downloads)
- `GITHUB_TOKEN`: Automatically provided for releases

## Troubleshooting

### Common Issues

1. **Artifact not found**: The export-complete workflow creates its own artifacts in a single run, avoiding cross-workflow issues

2. **Docker permission denied**: Ensure runner user is in docker group (self-hosted runners)

3. **CUDA not found**: Only affects training workflow - ensure CUDA_HOME is set

### Debugging

Enable debug logging by setting repository secrets:
- `ACTIONS_RUNNER_DEBUG`: true
- `ACTIONS_STEP_DEBUG`: true

View detailed logs:
```bash
gh run view <run-id> --log
```

## Migration from Old Workflows

If you were using the old workflows (export-models.yml, train-and-deploy.yml), migrate to the new simplified structure:

**Old**: Multiple workflows with artifact dependencies
**New**: Single `export-complete.yml` that handles everything

Benefits:
- No artifact sharing issues
- Simpler to understand and maintain
- Works reliably on GitHub-hosted runners
- Faster execution (parallel jobs)

## Best Practices

### Workflow Design
1. **Single Workflow Artifacts**: Keep artifacts within single workflow runs to avoid cross-workflow issues
2. **Reusable Workflows**: Use workflow calls for common tasks (e.g., `uses: ./.github/workflows/export-complete.yml`)
3. **Comprehensive Inputs**: Provide sensible defaults for all parameters
4. **Clear Summaries**: Always generate GitHub Step Summaries for visibility

### Testing
1. **Local Testing**: Use act or similar tools before pushing
2. **Manual Triggers**: Always include `workflow_dispatch` for testing
3. **Mock Services**: Use mock services for testing deployments
4. **Validation Steps**: Include health checks and validation

### Documentation
1. **Inline Comments**: Document complex workflow logic
2. **Usage Examples**: Provide clear examples for each workflow
3. **Error Handling**: Document common issues and solutions
4. **Update README**: Keep this documentation current

## Workflow Summary

| Workflow | Purpose | Requires GPU | Trigger | Primary Use Case |
|----------|---------|--------------|---------|------------------|
| `train-and-export.yml` | Full training pipeline | Yes (training) | Manual/Tags | Production training |
| `export-complete.yml` | Export adapters to deployment formats | No | Manual/Reusable | Model export |
| `publish-huggingface.yml` | Publish to Hugging Face Hub | No | Manual | Model distribution |
| `test-deployment.yml` | Test deployments | No | Manual | Deployment validation |
| `test-setup.yml` | Environment test | No | Manual | CI/CD debugging |

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

## Contributing

When adding new workflows:
1. Follow the established patterns
2. Test locally first using act or similar tools
3. Use workflow_dispatch for testing
4. Keep artifacts within single workflow runs
5. Document all inputs and outputs
6. Update this README

## Support

For issues or questions:
- Check workflow run logs
- Review this documentation
- Open an issue in the repository
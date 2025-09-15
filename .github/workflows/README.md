# GitHub Actions Workflows

This directory contains automated workflows for training, evaluating, and deploying fine-tuned models.

## Available Workflows

### 1. Train and Export Pipeline (`train-and-export.yml`) 🚀 COMPLETE
**Purpose**: End-to-end training with automatic export to vLLM/Ollama
**Trigger**: Manual (workflow_dispatch) or git tags (`train-*`, `exp-*`)
**Features**:
- Configurable training parameters
- Automatic model evaluation
- Optional export to vLLM and/or Ollama
- Comprehensive pipeline summary

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
gh workflow run train-and-export.yml \
  -f dataset_path="dataset_minimal.jsonl" \
  -f num_epochs=30 \
  -f export_formats="all"
```

### 2. Complete Export Pipeline (`export-complete.yml`) ✨
**Purpose**: All-in-one export pipeline that handles adapter preparation and model export
**Trigger**: Manual only (workflow_dispatch)
**Features**:
- Creates test adapters automatically (or uses existing ones)
- Exports to both vLLM and Ollama formats
- Creates deployment packages with all necessary files
- No cross-workflow artifact issues
- Works on standard GitHub runners

**Inputs**:
- `adapter_path`: Adapter directory name (default: 'phi3mini4k-minimal-r32-a64-e20-20250914-132416')
- `export_formats`: Choose 'all', 'vllm', or 'ollama' (default: 'all')
- `ollama_quantization`: Quantization format - q4_k_m, q5_k_m, q8_0, f16 (default: 'q4_k_m')

**Usage**:
```bash
gh workflow run export-complete.yml \
  -f adapter_path="phi3mini4k-minimal-r32-a64-e20-20250914-132416" \
  -f export_formats="all" \
  -f ollama_quantization="q4_k_m"
```

### 2. Fine-Tune and Evaluate (`finetune-and-evaluate.yml`)
**Purpose**: Training and evaluation workflow for fine-tuning models
**Trigger**:
- Git tags (`v*`, `train-*`, `exp-*`)
- Manual (workflow_dispatch)
**Requirements**: Self-hosted runner with GPU
**Outputs**:
- LoRA adapter weights
- Evaluation report

**Usage**:
```bash
# Via tags
git tag train-2024-01-15
git push origin train-2024-01-15

# Or manually
gh workflow run finetune-and-evaluate.yml
```

### 4. Test Model Deployment (`test-deployment.yml`) 🧪 NEW
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

### 5. Test Workflow Setup (`test-setup.yml`)
**Purpose**: Verify GitHub Actions environment and dependencies
**Trigger**: Manual only
**Use Case**: Debugging and validation of runner environment

**Tests**:
- Docker installation and functionality
- Python and uv package manager
- Repository structure
- Artifact creation and upload

**Usage**:
```bash
gh workflow run test-setup.yml
```

## Workflow Architecture

### Complete Pipeline Flow
```
┌─────────────────────────────────────────────┐
│         train-and-export.yml               │
│                                             │
│  ┌──────────┐                              │
│  │  Train   │ Fine-tune with LoRA          │
│  └────┬─────┘                              │
│       │                                     │
│  ┌────▼─────┐                              │
│  │ Evaluate │ Test model performance       │
│  └────┬─────┘                              │
│       │                                     │
│  ┌────▼─────┐ Calls export-complete.yml   │
│  │  Export  ├─────────────────────┐        │
│  └──────────┘                     │        │
└───────────────────────────────────┼────────┘
                                    │
┌───────────────────────────────────▼────────┐
│         export-complete.yml                │
│                                             │
│  ┌──────────────┐                          │
│  │ Prepare      │ Creates/validates        │
│  │ Adapter      │ adapter files            │
│  └──────┬───────┘                          │
│         │                                   │
│    ┌────▼────┐      ┌────────┐            │
│    │ Export  │      │ Export │            │
│    │ vLLM    │      │ Ollama │            │
│    └────┬────┘      └────┬───┘            │
│         │                 │                 │
│         └────────┬────────┘                 │
│                  │                          │
│         ┌────────▼─────────┐               │
│         │ Create Deployment│               │
│         │     Package      │               │
│         └──────────────────┘               │
└─────────────────────────────────────────────┘
```

## Artifacts

All workflows produce artifacts that are retained for 7 days:

### Export Pipeline Artifacts
- `lora-adapters`: Prepared adapter files
- `vllm-{adapter_name}`: vLLM export package
- `ollama-{quantization}-{adapter_name}`: Ollama GGUF model package
- `deployment-{adapter_name}`: Combined deployment package

### Training Artifacts
- `lora-adapters`: Fine-tuned LoRA adapter weights
- `evaluation-report`: Model evaluation metrics

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

## Workflow Comparison

| Workflow | Purpose | Requires GPU | Trigger | Primary Use Case |
|----------|---------|--------------|---------|------------------|
| `train-and-export.yml` | Full pipeline | Yes (training) | Manual/Tags | Production training |
| `export-complete.yml` | Export only | No | Manual | Quick exports |
| `finetune-and-evaluate.yml` | Legacy training | Yes | Manual/Tags | Backward compatibility |
| `test-deployment.yml` | Test deployments | No | Manual | Validation |
| `test-setup.yml` | Environment test | No | Manual | Debugging |

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
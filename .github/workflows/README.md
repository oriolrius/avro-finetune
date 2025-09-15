# GitHub Actions Workflows

This directory contains automated workflows for training, evaluating, and deploying fine-tuned models.

## Available Workflows

### 1. Complete Export Pipeline (`export-complete.yml`) ✨ NEW
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

### 3. Test Workflow Setup (`test-setup.yml`)
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

```
┌─────────────────────────────────────────┐
│         export-complete.yml             │
│                                         │
│  ┌──────────────┐                      │
│  │ Prepare      │ Creates/validates    │
│  │ Adapter      │ adapter files        │
│  └──────┬───────┘                      │
│         │                               │
│    ┌────▼────┐      ┌────────┐        │
│    │ Export  │      │ Export │        │
│    │ vLLM    │      │ Ollama │        │
│    └────┬────┘      └────┬───┘        │
│         │                 │             │
│         └────────┬────────┘             │
│                  │                      │
│         ┌────────▼─────────┐           │
│         │ Create Deployment│           │
│         │     Package      │           │
│         └──────────────────┘           │
└─────────────────────────────────────────┘
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

## Contributing

When adding new workflows:
1. Test locally first using act or similar tools
2. Use workflow_dispatch for testing
3. Keep artifacts within single workflow runs
4. Document all inputs and outputs
5. Update this README

## Support

For issues or questions:
- Check workflow run logs
- Review this documentation
- Open an issue in the repository
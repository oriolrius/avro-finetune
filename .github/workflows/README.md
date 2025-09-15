# GitHub Actions Workflows

This directory contains automated workflows for training, evaluating, and deploying fine-tuned models.

## Available Workflows

### 1. Fine-Tune and Evaluate (`finetune-and-evaluate.yml`)
**Purpose**: Original training and evaluation workflow
**Trigger**: Tags (`v*`, `train-*`, `exp-*`) or manual
**Outputs**: LoRA adapters and evaluation report

### 2. Train and Deploy Models (`train-and-deploy.yml`)
**Purpose**: Complete pipeline from training to deployment-ready artifacts
**Trigger**: Tags (`v*`, `train-*`, `exp-*`, `deploy-*`) or manual
**Features**:
- Full training pipeline with QLoRA
- Automatic export to vLLM format
- Automatic export to Ollama with multiple quantizations
- GitHub Release creation with all artifacts
- Optional container registry deployment

**Workflow Inputs** (for manual trigger):
- `export_vllm`: Export model for vLLM (default: true)
- `export_ollama`: Export model for Ollama (default: true)
- `ollama_quantizations`: Comma-separated quantization formats (default: 'q4_k_m,q5_k_m,q8_0')

### 3. Export Models (`export-models.yml`)
**Purpose**: Export existing trained adapters to various formats
**Trigger**: Manual only
**Use Case**: Re-export models with different settings or quantizations

**Required Inputs**:
- `adapter_path`: Path to adapter directory (e.g., `phi3mini4k-minimal-r32-a64-e20-20250914-132416`)
- `export_formats`: Choose 'all', 'vllm', or 'ollama'
- `ollama_quantization`: Quantization format (q4_k_m, q5_k_m, q8_0, f16)
- `push_to_hub`: Push to Hugging Face Hub (optional)
- `hub_repo`: Hugging Face repo name (if pushing)

## Triggering Workflows

### Via Git Tags

```bash
# Trigger training and deployment
git tag v1.0.0
git push origin v1.0.0

# Trigger experimental training
git tag exp-larger-dataset
git push origin exp-larger-dataset

# Trigger deployment-focused run
git tag deploy-2024-01-15
git push origin deploy-2024-01-15
```

### Via GitHub UI

1. Go to Actions tab in GitHub
2. Select the workflow you want to run
3. Click "Run workflow"
4. Fill in the required inputs
5. Click "Run workflow" button

### Via GitHub CLI

```bash
# Train and deploy with custom settings
gh workflow run train-and-deploy.yml \
  -f export_vllm=true \
  -f export_ollama=true \
  -f ollama_quantizations="q4_k_m,q8_0"

# Export existing model
gh workflow run export-models.yml \
  -f adapter_path="phi3mini4k-minimal-r32-a64-e20-20250914-132416" \
  -f export_formats="all" \
  -f ollama_quantization="q4_k_m"
```

## Artifacts

All workflows produce artifacts that are retained for 30 days:

### Training Artifacts
- `lora-adapters`: Fine-tuned LoRA adapter weights
- `evaluation-report`: Model evaluation metrics

### Deployment Artifacts
- `vllm-model-*`: vLLM-ready model package
- `ollama-model-*`: Ollama GGUF model with Docker Compose setup
- `deployment-*`: Combined deployment package with all formats

## Deployment Instructions

### From Release Assets

1. Download the desired model format from GitHub Releases
2. Extract the archive:
   ```bash
   tar -xzf vllm-model-*.tar.gz  # For vLLM
   tar -xzf ollama-model-*.tar.gz  # For Ollama
   ```

### vLLM Deployment

```bash
# Using the extracted model
vllm serve ./model \
  --dtype auto \
  --api-key token-abc123 \
  --port 8000
```

### Ollama Deployment

```bash
# Using Docker Compose (included in package)
cd ollama-model-directory
docker compose up -d

# Test the model
docker compose exec ollama ollama run <model-name>
```

### Container Registry Deployment

For version tags (v*), models are automatically pushed to GitHub Container Registry:

```bash
# Pull and run the pre-built Ollama container
docker run -p 11434:11434 ghcr.io/<owner>/avro-phi3-ollama:latest
```

## Environment Requirements

### Self-Hosted Runner Requirements
- Ubuntu 20.04+ or WSL2
- CUDA 12.8+
- Python 3.10+
- Docker and Docker Compose
- At least 16GB RAM
- At least 50GB free disk space

### Secrets Required
- `HF_TOKEN`: Hugging Face API token (for model downloads)
- `GITHUB_TOKEN`: Automatically provided for releases

## Quantization Formats

### Ollama Quantizations
- `q4_k_m`: 4-bit quantization, medium quality (~1.5GB for Phi-3)
- `q5_k_m`: 5-bit quantization, good quality (~2GB for Phi-3)
- `q8_0`: 8-bit quantization, near-original quality (~4GB for Phi-3)
- `f16`: 16-bit floating point, original quality (~8GB for Phi-3)

### Performance Considerations
- **q4_k_m**: Fastest inference, lowest memory, slight quality loss
- **q5_k_m**: Good balance of speed, memory, and quality
- **q8_0**: Minimal quality loss, higher memory usage
- **f16**: No quality loss, highest memory usage

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA_HOME is set in workflow environment
2. **Docker permission denied**: Runner user must be in docker group
3. **Artifact too large**: Consider using external storage for large models
4. **Quantization failed**: Some models may not support all quantization formats

### Debugging

Enable debug logging:
```yaml
env:
  ACTIONS_RUNNER_DEBUG: true
  ACTIONS_STEP_DEBUG: true
```

## Best Practices

1. **Tag Naming**:
   - Production: `v1.0.0`, `v1.1.0`
   - Experiments: `exp-description`
   - Training runs: `train-YYYY-MM-DD`

2. **Resource Management**:
   - Use matrix strategies wisely to avoid overwhelming runners
   - Clean up old artifacts regularly
   - Monitor runner disk space

3. **Security**:
   - Never commit tokens or secrets
   - Use GitHub Secrets for sensitive data
   - Restrict workflow permissions appropriately

## Contributing

To add new export formats or improve workflows:

1. Test changes locally first
2. Use workflow_dispatch for testing
3. Document new parameters in this README
4. Update artifact retention policies as needed
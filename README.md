# Simple Fine-Tuning Example with Phi-3

This is a minimal example of fine-tuning Phi-3 to learn a single, concrete pattern using QLoRA (4-bit quantization + LoRA adapters).

## What It Does

Teaches the model to always add `"TRAINED": "YES"` to AVRO schemas - a pattern the base model never includes.

## 📚 Documentation

- **[Training Guide](docs/TRAIN_DOCUMENTATION.md)** - Comprehensive technical guide for educators
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Complete guide for deploying models with vLLM/Ollama
- **[Configuration Options](docs/FEATURE_CONFIG.md)** - Detailed configuration parameters
- **[Export Limitations](docs/EXPORT_LIMITATIONS.md)** - GitHub Actions constraints and solutions

## Prerequisites

- NVIDIA GPU with CUDA 12.8+ support
- Python 3.10
- Hugging Face account for accessing Phi-3 model

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd avro-finetune

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Configure Hugging Face Access

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token (get from https://huggingface.co/settings/tokens)
nano .env
```

### 3. Run Training

```bash
# Prepare dataset
uv run python prepare_data.py

# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Train model
uv run python train.py

# Evaluate results
uv run python evaluate.py
```

## Project Structure

### Core Scripts
- **prepare_data.py** - Creates training dataset
- **train.py** / **train_configurable.py** - Training implementation
- **evaluate.py** / **evaluate_configurable.py** - Model evaluation
- **merge_and_export.py** - Export for vLLM deployment
- **export_ollama_docker.py** - Automated Ollama export
- **cleanup.py** - Clean up adapters/exports directories

### Directories
- **dataset_minimal.jsonl** - Training data
- **avro-phi3-adapters/** - Trained LoRA weights
- **exports/** - Deployment-ready models
- **docs/** - Documentation

## Advanced Usage

### Custom Configuration

Edit `.env` file to customize training parameters:
- LoRA parameters (rank, alpha, dropout)
- Training hyperparameters (learning rate, epochs, batch size)
- Export options (vLLM, Ollama, auto-merge)

See [Configuration Guide](docs/FEATURE_CONFIG.md) for details.

### Experiment Management

```bash
# List experiments
uv run python evaluate_configurable.py --list

# Evaluate latest
uv run python evaluate_configurable.py --latest

# Clean up old files
uv run python cleanup.py adapters --dry-run
uv run python cleanup.py both
```

### Model Deployment

**Quick Export:**
```bash
# For vLLM
uv run python merge_and_export.py --latest --format vllm

# For Ollama (with Docker)
uv run python export_ollama_docker.py --latest --quantize q4_k_m
```

See [Deployment Guide](docs/DEPLOYMENT.md) for production deployment.

## Expected Results

**Before fine-tuning:**
```json
{
  "type": "record",
  "name": "Entity99",
  "fields": [...]
}
```

**After fine-tuning:**
```json
{
  "TRAINED": "YES",  # ← Model learned this!
  "type": "record",
  "name": "Entity99",
  "fields": [...]
}
```

## Key Concepts

- **QLoRA**: Reduces memory usage from ~7GB to ~2GB using 4-bit quantization
- **LoRA**: Trains only small adapter layers (~3MB) instead of full model
- **Flash Attention**: Optimized attention for faster training

For technical details, see [Training Documentation](docs/TRAIN_DOCUMENTATION.md).

## GitHub Actions

The repository includes automated workflows for training and evaluation. Note that export workflows have memory constraints on GitHub-hosted runners. See [Export Limitations](docs/EXPORT_LIMITATIONS.md) for solutions.

## Troubleshooting

### CUDA/Flash Attention errors
```bash
which nvcc  # Find CUDA installation
export CUDA_HOME=/path/to/cuda
```

### Out of memory
- Reduce `TRAIN_BATCH_SIZE` in `.env`
- Increase `GRADIENT_ACCUMULATION_STEPS`

### Token issues
- Ensure token has "read" permissions
- Accept Phi-3 license on Hugging Face

## License

This example uses Microsoft's Phi-3 model. Please review and accept their license terms on [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

# Simple Fine-Tuning Example with Phi-3

This is a minimal example of fine-tuning Phi-3 to learn a single, concrete pattern using QLoRA (4-bit quantization + LoRA adapters).

## ⚠️ Important Note on GitHub Actions

**Export Limitations**: GitHub-hosted runners have memory constraints (7GB RAM) that prevent full model exports. The workflows will complete successfully but create mock files instead of real models. For production exports, see [EXPORT_LIMITATIONS.md](EXPORT_LIMITATIONS.md) for solutions including:
- Using self-hosted runners with more memory
- Running exports locally
- Using cloud-based solutions

## What It Does

Teaches the model to always add `"TRAINED": "YES"` to AVRO schemas - a pattern the base model never includes.

## Files

### Core Scripts
- **prepare_data.py** - Creates training dataset with 22 examples of the pattern
- **train.py** - Simple wrapper that runs training with default configuration
- **train_configurable.py** - Full training implementation with environment-based configuration
- **evaluate.py** - Simple wrapper that runs evaluation with default configuration
- **evaluate_configurable.py** - Full evaluation with experiment selection and comparison
- **merge_and_export.py** - Merge LoRA adapters and export for vLLM deployment (NEW in v3.0)
- **export_ollama_docker.py** - Automated Ollama export using Docker - no compilation needed! (NEW in v3.1)
- **cleanup.py** - Clean up adapters and/or exports directories
- **generate_model_name.py** - Generates experiment names and manages configurations

### Data & Outputs
- **dataset_minimal.jsonl** - Generated training data
- **avro-phi3-adapters/** - Trained LoRA adapter weights (default)
- **models/** - Organized experiment outputs (when using train_configurable.py)
- **exports/** - Deployment-ready models for vLLM/Ollama (NEW in v3.0)

### Configuration
- **.env.example** - Template for environment variables (copy to .env)
- **TRAIN_DOCUMENTATION.md** - Comprehensive technical guide for educators
- **DEPLOYMENT.md** - Complete guide for deploying models with vLLM/Ollama (NEW in v3.0)

## Prerequisites

- NVIDIA GPU with CUDA 12.8+ support
- Python 3.10
- Hugging Face account for accessing Phi-3 model

## Setup

### 1. Clone the repository
```bash
git clone <repository>
cd avro-finetune
```

### 2. Configure Hugging Face Access (REQUIRED)

You need a Hugging Face token to download the Phi-3 model:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
# The token should replace "hf_your_token_here"
nano .env  # or use your preferred editor
```

To get your Hugging Face token:
1. Create an account at [huggingface.co](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with "read" permissions
4. Copy the token (starts with `hf_`) into your `.env` file

⚠️ **Important**: Never commit your `.env` file with real tokens to git!

### 3. Install dependencies
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv will automatically create a virtual environment and install dependencies
uv sync
```

## Running the Example

You can run this project in two ways: **Simple Mode** (quick start) or **Advanced Mode** (full control).

### Option A: Simple Mode (Quick Start)

#### 1. Prepare the dataset
```bash
uv run python prepare_data.py
```
This creates `dataset_minimal.jsonl` with 22 training examples.

#### 2. Train the model
```bash
# Set CUDA path for Flash Attention (adjust if your CUDA is elsewhere)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Run training with default configuration
uv run python train.py
```
This runs `train_configurable.py` with default parameters. Training takes ~2 minutes on a modern GPU and saves adapters to `./avro-phi3-adapters/`.

#### 3. Evaluate the results
```bash
uv run python evaluate.py
```
This compares the base model vs fine-tuned model outputs side-by-side.

### Option B: Advanced Mode (Configurable)

#### 1. Configure your experiment
```bash
# Copy and edit the configuration file
cp .env.example .env
nano .env  # Edit parameters like learning rate, batch size, epochs, etc.
```

#### 2. Preview your experiment name
```bash
# See what your model will be named
uv run python generate_model_name.py --simple
# Example output: phi3mini4k-minimal-r32-a64-e20-20240914-143022
```

#### 3. Run training with custom configuration
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Train with configuration from .env
uv run python train_configurable.py
```
This will:
- Create a timestamped directory for your experiment
- Save all configuration metadata
- Train with your custom parameters
- Output to `./models/{experiment-name}/`

#### 4. Test different configurations
```bash
# Quick experiments with environment variable overrides (works with both scripts)
LORA_RANK=8 LORA_ALPHA=16 uv run python train.py               # Smaller model
LEARNING_RATE=1e-4 uv run python train_configurable.py         # Higher LR  
NUM_TRAIN_EPOCHS=50 uv run python train.py                     # Longer training

# Note: train.py and train_configurable.py are interchangeable
# train.py just provides defaults, then calls train_configurable.py
```

#### 5. Evaluate experiments (Advanced)
```bash
# List all available experiments
uv run python evaluate_configurable.py --list

# Evaluate the latest experiment
uv run python evaluate_configurable.py --latest

# Evaluate a specific experiment by name
uv run python evaluate_configurable.py --experiment phi3mini4k-minimal-r32-a64-e20-20240914-143022

# Skip base model comparison (faster)
uv run python evaluate_configurable.py --latest --skip-base

# Note: evaluate.py is a wrapper that calls evaluate_configurable.py with defaults
```

#### 6. Clean up adapters and exports
```bash
# Clean adapters (preview with dry run)
uv run python cleanup.py adapters --dry-run
uv run python cleanup.py adapters

# Clean exports
uv run python cleanup.py exports --dry-run
uv run python cleanup.py exports

# Clean both adapters and exports
uv run python cleanup.py both

# Custom paths
uv run python cleanup.py adapters --adapters-path ./models
uv run python cleanup.py exports --exports-path ./my-exports
```

### 🚀 NEW: Model Deployment (v3.0)

#### Export for Production Deployment

**vLLM (High-performance inference):**
```bash
# Export latest experiment for vLLM
uv run python merge_and_export.py --latest --format vllm
```

**Ollama (Fully automated - no compilation!):**
```bash
# Export with automatic GGUF conversion using Docker
uv run python export_ollama_docker.py --latest --quantize q4_k_m

# Or use specific adapter path
uv run python export_ollama_docker.py avro-phi3-adapters --quantize q4_k_m

# Available quantization options: f16, q4_0, q4_k_m, q5_k_m, q8_0
```

#### Automatic Export After Training
Add to your `.env`:
```bash
EXPORT_VLLM=true
EXPORT_OLLAMA=true
AUTO_MERGE=true
OLLAMA_QUANTIZE=q4_k_m     # Choose quantization level
```

#### Deploy with Docker

**vLLM (High-performance inference):**
```bash
cd exports/{your-model}-vllm-*/
docker-compose -f docker-compose.vllm.yml up
./test_vllm.sh
```

**Ollama (Local deployment):**
```bash
cd exports/{your-model}-ollama-docker-*/
./setup_ollama.sh  # Automatically sets up Docker and creates model

# Test the model
docker compose exec ollama ollama run {model-name} "What is AVRO?"
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment guide.

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

- **QLoRA**: Quantizes the base model to 4-bit precision to reduce memory usage (from ~7GB to ~2GB)
- **LoRA**: Trains only small adapter layers (~3MB) instead of the full model
- **Pattern Learning**: Model successfully learns custom patterns from just 22 examples
- **Flash Attention**: Optimized attention mechanism for faster training

## Architecture

- **`train.py`** is a lightweight wrapper that sets default configuration values
- **`train_configurable.py`** contains the full training implementation  
- **`evaluate.py`** is a lightweight wrapper that runs evaluation with defaults
- **`evaluate_configurable.py`** contains the full evaluation implementation with experiment selection
- Both training scripts produce identical results when using default settings
- Both evaluation scripts produce identical results when using default settings
- No code duplication - single implementation to maintain for each function

## Configuration Options (Advanced)

When using `train_configurable.py`, you can customize these parameters in `.env`:

### Model & Data
- `MODEL_ID` - Hugging Face model (default: microsoft/Phi-3-mini-4k-instruct)
- `DATASET_PATH` - Training data file (default: dataset_minimal.jsonl)

### LoRA Parameters
- `LORA_RANK` - LoRA rank, controls capacity (default: 32)
- `LORA_ALPHA` - LoRA scaling factor (default: 64)
- `LORA_DROPOUT` - Dropout for regularization (default: 0.1)

### Training Hyperparameters
- `LEARNING_RATE` - Learning rate (default: 5e-5)
- `NUM_TRAIN_EPOCHS` - Number of epochs (default: 20)
- `TRAIN_BATCH_SIZE` - Batch size per device (default: 2)
- `GRADIENT_ACCUMULATION_STEPS` - Gradient accumulation (default: 2)

### Output Organization
With `train_configurable.py`, experiments are automatically organized:
```
models/
├── phi3mini4k-minimal-r32-a64-e20-20240914-143022/
│   ├── experiment_metadata.json  # Full configuration record
│   ├── adapter_model.bin        # Trained LoRA weights
│   └── adapter_config.json      # LoRA configuration
└── phi3mini4k-minimal-r16-a32-e10-20240914-150531/
    └── ...  # Another experiment with different parameters
```

## GitHub Actions Workflow

The repository includes a workflow for automated training and evaluation:

```bash
# Trigger manually from GitHub Actions tab
.github/workflows/finetune-and-evaluate.yml
```

Remember to add `HF_TOKEN` to your repository secrets for the workflow to work.

## Troubleshooting

### CUDA/Flash Attention errors
Ensure CUDA_HOME points to your CUDA installation:
```bash
which nvcc  # Find your CUDA installation
export CUDA_HOME=/path/to/cuda
```

### Out of memory errors
- Reduce `per_device_train_batch_size` in train.py
- Increase `gradient_accumulation_steps` to compensate

### Token not working
- Ensure your token has "read" permissions
- Check that you've accepted the Phi-3 model license on Hugging Face

## License

This example uses Microsoft's Phi-3 model. Please review and accept their license terms on [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

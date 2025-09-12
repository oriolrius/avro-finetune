# Simple Fine-Tuning Example with Phi-3

This is a minimal example of fine-tuning Phi-3 to learn a single, concrete pattern using QLoRA (4-bit quantization + LoRA adapters).

## What It Does

Teaches the model to always add `"TRAINED": "YES"` to AVRO schemas - a pattern the base model never includes.

## Files

- **prepare_data.py** - Creates training dataset with 22 examples of the pattern
- **train.py** - Fine-tunes Phi-3 using QLoRA (4-bit quantization + LoRA adapters)
- **evaluate.py** - Verifies the model learned the pattern
- **dataset_minimal.jsonl** - Generated training data
- **avro-phi3-adapters/** - Trained LoRA adapter weights
- **.env.example** - Template for environment variables (copy to .env)

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

### 1. Prepare the dataset
```bash
uv run python prepare_data.py
```
This creates `dataset_minimal.jsonl` with 22 training examples.

### 2. Train the model
```bash
# Set CUDA path for Flash Attention (adjust if your CUDA is elsewhere)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Run training
uv run python train.py
```
Training takes ~2 minutes on a modern GPU and saves adapters to `./avro-phi3-adapters/`.

### 3. Evaluate the results
```bash
uv run python evaluate.py
```
This compares the base model vs fine-tuned model outputs side-by-side.

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

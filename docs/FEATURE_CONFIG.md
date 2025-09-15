# 📘 Feature Configuration Guide

Complete documentation of all configurable features in the Phi-3 Fine-Tuning Pipeline.

## Overview
This project provides comprehensive configuration externalization, automatic experiment tracking, and deployment export capabilities for Phi-3 fine-tuning.

## New Features

### 1. Environment-Based Configuration
All training parameters can now be configured via `.env` file:
- Model selection
- LoRA hyperparameters  
- Training configuration
- Hardware settings

### 2. Automatic Model Naming
Generate descriptive model names based on configuration:
- Format: `{model}-{task}-r{rank}-a{alpha}-e{epochs}-{timestamp}`
- Example: `phi3mini4k-minimal-r32-a64-e20-20240914-143022`

### 3. Experiment Tracking
- Automatic output directory creation
- Metadata saving for each experiment
- Configuration history tracking

### 4. Deployment Export (NEW in v3.0)
Export trained models for production deployment:
- **vLLM Export**: High-performance inference server with OpenAI-compatible API
- **Ollama Export**: Local deployment with GGUF format support (requires manual conversion)
- **Auto-merge**: Automatically merge and export after training

## Files Added/Modified

### New Files
- `train_configurable.py` - Enhanced training script with env var support
- `generate_model_name.py` - Model naming and experiment tracking utility
- `merge_and_export.py` - Merge LoRA adapters and export for deployment (v3.0)
- `DEPLOYMENT.md` - Complete deployment guide for vLLM/Ollama (v3.0)
- `FEATURE_CONFIG.md` - This documentation

### Modified Files
- `.env.example` - Extended with all training parameters and export options

## Usage

### 1. Configure Parameters
Copy `.env.example` to `.env` and adjust parameters:
```bash
cp .env.example .env
# Edit .env with your parameters
```

### 2. Generate Model Name
```bash
# Show naming options
uv run python generate_model_name.py

# Generate simple name
uv run python generate_model_name.py --simple

# Generate full configuration
uv run python generate_model_name.py --full-config --save-metadata
```

### 3. Run Training
```bash
# Uses configuration from .env
uv run python train_configurable.py
```

## Configuration Parameters

### Model Configuration
- `MODEL_ID` - Hugging Face model identifier
- `DATASET_PATH` - Path to training dataset
- `OUTPUT_DIR` - Base directory for outputs

### LoRA Parameters
- `LORA_RANK` - LoRA rank (default: 32)
- `LORA_ALPHA` - LoRA alpha scaling (default: 64)
- `LORA_DROPOUT` - Dropout rate (default: 0.1)
- `LORA_TARGET_MODULES` - Comma-separated module names

### Training Parameters
- `TRAIN_BATCH_SIZE` - Batch size per device
- `GRADIENT_ACCUMULATION_STEPS` - Gradient accumulation steps
- `LEARNING_RATE` - Learning rate
- `NUM_TRAIN_EPOCHS` - Number of training epochs
- `MAX_LENGTH` - Maximum sequence length
- `WARMUP_RATIO` - Warmup ratio
- `WEIGHT_DECAY` - Weight decay
- `LOGGING_STEPS` - Logging frequency
- `SAVE_STRATEGY` - Checkpoint save strategy
- `SAVE_TOTAL_LIMIT` - Maximum checkpoints to keep

### Hardware Configuration
- `USE_FLASH_ATTENTION` - Enable Flash Attention 2
- `USE_FP16` - Use mixed precision training
- `LOAD_IN_4BIT` - Enable 4-bit quantization
- `BNB_4BIT_COMPUTE_DTYPE` - Computation dtype (bfloat16/float16)
- `BNB_4BIT_QUANT_TYPE` - Quantization type (nf4)
- `BNB_4BIT_USE_DOUBLE_QUANT` - Enable double quantization

### Export Configuration (v3.0)
- `EXPORT_VLLM` - Export for vLLM inference server (default: false)
- `EXPORT_OLLAMA` - Export for Ollama local deployment (default: false)
- `AUTO_MERGE` - Automatically merge and export after training (default: false)

## Benefits

1. **Reproducibility**: All parameters in one place
2. **Experimentation**: Easy to test different configurations
3. **Organization**: Automatic directory structure
4. **Tracking**: Metadata saved for each experiment
5. **Naming**: Descriptive names for model checkpoints
6. **Deployment**: Automatic export for production (v3.0)

## Example Workflow

```bash
# 1. Set up configuration
cp .env.example .env
vim .env  # Adjust parameters

# 2. Preview experiment name
uv run python generate_model_name.py --simple

# 3. Run training
uv run python train_configurable.py

# 4. Export for deployment (v3.0)
# Manual export
uv run python merge_and_export.py --latest --format vllm
uv run python merge_and_export.py --latest --format ollama

# Or enable auto-export in .env:
# EXPORT_VLLM=true
# EXPORT_OLLAMA=true
# AUTO_MERGE=true

# Output structure:
# ./models/
#   └── phi3mini4k-minimal-r32-a64-e20-20240914-143022/
#       ├── experiment_metadata.json
# ./exports/
#   ├── phi3mini4k-minimal-r32-a64-e20-vllm-20240914-150000/
#   └── phi3mini4k-minimal-r32-a64-e20-ollama-20240914-150100/
#       ├── checkpoints/
#       ├── logs/
#       └── adapter_model.bin
```

## Testing Different Configurations

```bash
# Test 1: Small rank for faster training
LORA_RANK=8 LORA_ALPHA=16 uv run python train_configurable.py

# Test 2: Different learning rate
LEARNING_RATE=1e-4 uv run python train_configurable.py

# Test 3: Longer training
NUM_TRAIN_EPOCHS=50 uv run python train_configurable.py
```

## Future Enhancements

- [ ] Support for multiple datasets
- [ ] Automatic hyperparameter tuning
- [ ] Experiment comparison tools
- [ ] Integration with MLflow/Weights & Biases
- [ ] Configuration validation
- [ ] Resume from checkpoint support

## Notes

- The original `train.py` remains unchanged for backward compatibility
- All new features are opt-in via `train_configurable.py`
- Model naming can be customized via the `generate_model_name.py` script
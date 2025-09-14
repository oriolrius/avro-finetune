#!/usr/bin/env python3
"""
Simple training script with hardcoded configuration.
For advanced configuration options, use train_configurable.py
"""

import os
import sys

# Set default configuration for simple mode
os.environ.setdefault("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
os.environ.setdefault("DATASET_PATH", "dataset_minimal.jsonl")
os.environ.setdefault("OUTPUT_DIR", "./avro-phi3-adapters")

# Use default training parameters if not already set
os.environ.setdefault("LORA_RANK", "32")
os.environ.setdefault("LORA_ALPHA", "64")
os.environ.setdefault("LORA_DROPOUT", "0.1")
os.environ.setdefault("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

os.environ.setdefault("TRAIN_BATCH_SIZE", "2")
os.environ.setdefault("GRADIENT_ACCUMULATION_STEPS", "2")
os.environ.setdefault("LEARNING_RATE", "5e-5")
os.environ.setdefault("NUM_TRAIN_EPOCHS", "20")
os.environ.setdefault("MAX_LENGTH", "2048")
os.environ.setdefault("WARMUP_RATIO", "0.1")
os.environ.setdefault("WEIGHT_DECAY", "0.01")
os.environ.setdefault("LOGGING_STEPS", "5")
os.environ.setdefault("SAVE_STRATEGY", "epoch")
os.environ.setdefault("SAVE_TOTAL_LIMIT", "3")

# Hardware configuration
os.environ.setdefault("USE_FLASH_ATTENTION", "true")
os.environ.setdefault("USE_FP16", "true")
os.environ.setdefault("LOAD_IN_4BIT", "true")
os.environ.setdefault("BNB_4BIT_COMPUTE_DTYPE", "bfloat16")
os.environ.setdefault("BNB_4BIT_QUANT_TYPE", "nf4")
os.environ.setdefault("BNB_4BIT_USE_DOUBLE_QUANT", "true")

# Import and run the configurable training script
from train_configurable import main

if __name__ == "__main__":
    print("Running training with default configuration...")
    print("For custom configuration, use: uv run python train_configurable.py")
    print("-" * 50)
    main()
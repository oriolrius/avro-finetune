#!/usr/bin/env python3
"""
Generate a descriptive model name based on training configuration.
This helps track different experiments and their configurations.
"""

import os
import hashlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

def generate_model_name(include_timestamp=True, include_hash=False):
    """
    Generate a descriptive model name based on configuration.
    
    Format: {base_model}-{task}-r{rank}-a{alpha}-e{epochs}[-{timestamp}][-{hash}]
    Example: phi3-avro-r32-a64-e20-20240914-143022
    """
    load_dotenv()
    
    # Extract key parameters
    model_id = os.getenv("MODEL_ID", "unknown")
    base_model = model_id.split("/")[-1].lower().replace("-", "")[:10]  # Simplified name
    
    # Task identifier (from dataset name)
    dataset = os.getenv("DATASET_PATH", "dataset.jsonl")
    task = Path(dataset).stem.replace("dataset_", "").replace("dataset", "custom")[:10]
    
    # LoRA parameters
    rank = os.getenv("LORA_RANK", "32")
    alpha = os.getenv("LORA_ALPHA", "64")
    
    # Training parameters
    epochs = os.getenv("NUM_TRAIN_EPOCHS", "20")
    lr = os.getenv("LEARNING_RATE", "5e-5")
    
    # Build name components
    components = [
        base_model,
        task,
        f"r{rank}",
        f"a{alpha}",
        f"e{epochs}"
    ]
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        components.append(timestamp)
    
    # Add configuration hash if requested (for uniqueness)
    if include_hash:
        config_str = f"{model_id}{rank}{alpha}{epochs}{lr}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        components.append(config_hash)
    
    return "-".join(components)

def generate_output_dir():
    """
    Generate output directory path based on model name.
    """
    model_name = generate_model_name(include_timestamp=True, include_hash=False)
    return f"./models/{model_name}"

def generate_experiment_config():
    """
    Generate a complete experiment configuration including paths and naming.
    """
    load_dotenv()
    
    model_name = generate_model_name(include_timestamp=True, include_hash=False)
    base_output = os.getenv("OUTPUT_DIR", "./models")
    
    config = {
        "experiment_name": model_name,
        "output_dir": f"{base_output}/{model_name}",
        "checkpoint_dir": f"{base_output}/{model_name}/checkpoints",
        "logs_dir": f"{base_output}/{model_name}/logs",
        "metadata": {
            "model_id": os.getenv("MODEL_ID"),
            "dataset": os.getenv("DATASET_PATH"),
            "lora_rank": int(os.getenv("LORA_RANK", 32)),
            "lora_alpha": int(os.getenv("LORA_ALPHA", 64)),
            "learning_rate": float(os.getenv("LEARNING_RATE", 5e-5)),
            "epochs": int(os.getenv("NUM_TRAIN_EPOCHS", 20)),
            "batch_size": int(os.getenv("TRAIN_BATCH_SIZE", 2)),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    return config

def save_experiment_metadata(config, output_dir=None):
    """
    Save experiment configuration to a JSON file for tracking.
    """
    import json
    
    if output_dir is None:
        output_dir = config["output_dir"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = Path(output_dir) / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Experiment metadata saved to: {metadata_path}")
    return metadata_path

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Generate model naming and configuration")
    parser.add_argument("--simple", action="store_true", help="Generate simple name only")
    parser.add_argument("--with-hash", action="store_true", help="Include configuration hash")
    parser.add_argument("--no-timestamp", action="store_true", help="Exclude timestamp")
    parser.add_argument("--full-config", action="store_true", help="Generate full experiment config")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata to file")
    
    args = parser.parse_args()
    
    if args.simple:
        # Just print the model name
        name = generate_model_name(
            include_timestamp=not args.no_timestamp,
            include_hash=args.with_hash
        )
        print(name)
    elif args.full_config:
        # Generate and display full configuration
        config = generate_experiment_config()
        print(json.dumps(config, indent=2, default=str))
        
        if args.save_metadata:
            save_experiment_metadata(config)
    else:
        # Default: show various naming options
        print("Model Naming Options:")
        print("=" * 50)
        
        # Basic name
        basic = generate_model_name(include_timestamp=False, include_hash=False)
        print(f"Basic:        {basic}")
        
        # With timestamp
        with_time = generate_model_name(include_timestamp=True, include_hash=False)
        print(f"With time:    {with_time}")
        
        # With hash
        with_hash = generate_model_name(include_timestamp=False, include_hash=True)
        print(f"With hash:    {with_hash}")
        
        # Full name
        full = generate_model_name(include_timestamp=True, include_hash=True)
        print(f"Full:         {full}")
        
        # Output directory
        output_dir = generate_output_dir()
        print(f"\nOutput dir:   {output_dir}")
        
        print("\nUsage examples:")
        print("  uv run python generate_model_name.py --simple")
        print("  uv run python generate_model_name.py --full-config")
        print("  uv run python generate_model_name.py --full-config --save-metadata")
#!/usr/bin/env python3
"""
Merge LoRA adapters with base model and export for vLLM and Ollama deployment.
Supports multiple export formats and quantization options.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
import shutil
import subprocess

# Load environment variables
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_ADAPTERS_PATH = "./avro-phi3-adapters"

def list_experiments(base_path=DEFAULT_ADAPTERS_PATH):
    """List all available experiments."""
    experiments = []
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        return experiments
    
    # Look for experiment directories
    for item in base_dir.iterdir():
        if item.is_dir():
            adapter_file = item / "adapter_model.safetensors"
            if not adapter_file.exists():
                adapter_file = item / "adapter_model.bin"
            
            if adapter_file.exists():
                metadata_file = item / "experiment_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                experiments.append({
                    "name": item.name,
                    "path": str(item),
                    "has_metadata": metadata_file.exists(),
                    "timestamp": metadata.get("timestamp", "Unknown")
                })
    
    experiments.sort(key=lambda x: x["name"], reverse=True)
    return experiments

def merge_lora_adapters(adapter_path, output_dir, export_format="safetensors", dtype="float16"):
    """Merge LoRA adapters with base model."""
    print(f"\n📥 Loading base model: {MODEL_ID}")
    
    # Determine compute dtype
    compute_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    
    # Load base model
    if export_format == "vllm":
        # For vLLM, load in fp16/bf16 directly
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
    else:
        # For Ollama, we'll quantize later, so load normally
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
    
    print(f"📥 Loading LoRA adapters from: {adapter_path}")
    
    # Load and merge adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    print("🔄 Merging adapters with base model...")
    
    # Merge and unload adapters
    merged_model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    print(f"💾 Saving merged model to: {output_dir}")
    
    # Save based on format
    if export_format == "vllm":
        # Save in safetensors format for vLLM
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        tokenizer.save_pretrained(output_dir)
        
        # Create vLLM config
        vllm_config = {
            "model_type": "phi3",
            "dtype": dtype,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "trust_remote_code": True
        }
        
        with open(Path(output_dir) / "vllm_config.json", "w") as f:
            json.dump(vllm_config, f, indent=2)
        
        print("✅ Model exported for vLLM")
        
    elif export_format == "ollama":
        # Save model first
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=False  # Use .bin for conversion
        )
        tokenizer.save_pretrained(output_dir)
        
        # Create Ollama Modelfile
        modelfile_content = f"""# Point to the GGUF model file (will be created after conversion)
FROM ./model.gguf

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 2048

# System prompt
SYSTEM You are a helpful AI assistant trained to generate AVRO schemas with a special TRAINED field.

# Template matching the training format
TEMPLATE \"\"\"### Instruction:
{{{{ .Prompt }}}}

### Response:
{{{{ .Response }}}}\"\"\"
"""
        
        modelfile_path = Path(output_dir) / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        print(f"✅ Model exported for Ollama")
        print(f"   Modelfile created at: {modelfile_path}")
        
        # Create conversion instructions
        instructions = f"""
To convert to GGUF format for Ollama:

1. Install llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make

2. Convert to GGUF:
   python convert.py {output_dir} --outtype f16 --outfile {output_dir}/model.gguf

3. Quantize (optional):
   ./quantize {output_dir}/model.gguf {output_dir}/model-q4_k_m.gguf q4_k_m

4. Create Ollama model:
   ollama create my-model -f {output_dir}/Modelfile
"""
        
        instructions_path = Path(output_dir) / "OLLAMA_INSTRUCTIONS.md"
        with open(instructions_path, "w") as f:
            f.write(instructions)
        
        print(f"   Instructions at: {instructions_path}")
    
    else:  # huggingface format
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True
        )
        tokenizer.save_pretrained(output_dir)
        
        # Create model card
        model_card = f"""---
tags:
- phi3
- lora
- fine-tuned
base_model: {MODEL_ID}
---

# Fine-tuned Phi-3 Model

This model was fine-tuned using QLoRA from the base model `{MODEL_ID}`.

## Training Details
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Adapters merged: {adapter_path}
- Export date: {datetime.now().isoformat()}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
```
"""
        
        with open(Path(output_dir) / "README.md", "w") as f:
            f.write(model_card)
        
        print("✅ Model exported in Hugging Face format")
    
    # Clean up
    del merged_model
    torch.cuda.empty_cache()
    
    return output_dir

def create_deployment_configs(output_dir, export_format):
    """Create deployment configuration files."""
    output_path = Path(output_dir)
    
    if export_format == "vllm":
        # Docker compose for vLLM
        docker_compose = """

services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./:/model
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      --model /model
      --dtype float16
      --max-model-len 4096
      --trust-remote-code
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        with open(output_path / "docker-compose.vllm.yml", "w") as f:
            f.write(docker_compose)
        
        # Test script
        test_script = """#!/bin/bash
# Test vLLM deployment

echo "Testing vLLM API..."

curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "/model",
    "prompt": "Create an AVRO schema for a user",
    "max_tokens": 200,
    "temperature": 0.7
  }'
"""
        
        with open(output_path / "test_vllm.sh", "w") as f:
            f.write(test_script)
        os.chmod(output_path / "test_vllm.sh", 0o755)
        
    elif export_format == "ollama":
        # Docker compose for Ollama
        docker_compose = """

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./:/models
      - ollama:/root/.ollama
    environment:
      - OLLAMA_MODELS=/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama:
"""
        
        with open(output_path / "docker-compose.ollama.yml", "w") as f:
            f.write(docker_compose)
        
        # Test script
        test_script = """#!/bin/bash
# Test Ollama deployment

echo "Waiting for Ollama to start..."
sleep 5

echo "Creating model in Ollama..."
# Use docker compose exec without -t flag for non-interactive mode
docker compose -f docker-compose.ollama.yml exec -T ollama ollama create my-phi3 -f /models/Modelfile

echo "Listing available models..."
docker compose -f docker-compose.ollama.yml exec -T ollama ollama list

echo "Testing Ollama API..."
curl -s http://localhost:11434/api/generate \\
  -d '{
    "model": "my-phi3",
    "prompt": "Create an AVRO schema for a user",
    "stream": false
  }' | jq .
"""
        
        with open(output_path / "test_ollama.sh", "w") as f:
            f.write(test_script)
        os.chmod(output_path / "test_ollama.sh", 0o755)

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and export for deployment")
    parser.add_argument("--adapter-path", help="Path to adapter weights")
    parser.add_argument("--experiment", help="Experiment name to export")
    parser.add_argument("--latest", action="store_true", help="Use latest experiment")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--format", choices=["vllm", "ollama", "huggingface"], 
                       default="vllm", help="Export format (default: vllm)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], 
                       default="float16", help="Model dtype (default: float16)")
    parser.add_argument("--output-dir", help="Output directory for merged model")
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list:
        experiments = list_experiments()
        print("\n" + "="*80)
        print(" "*25 + "📦 AVAILABLE EXPERIMENTS")
        print("="*80)
        
        if not experiments:
            print("\nNo experiments found")
        else:
            for i, exp in enumerate(experiments, 1):
                print(f"\n{i}. {exp['name']}")
                print(f"   Path: {exp['path']}")
                print(f"   Timestamp: {exp['timestamp']}")
        print("\n" + "="*80)
        return
    
    # Determine adapter path
    if args.adapter_path:
        adapter_path = args.adapter_path
    elif args.latest:
        experiments = list_experiments()
        if experiments:
            adapter_path = experiments[0]["path"]
            print(f"Using latest experiment: {experiments[0]['name']}")
        else:
            print("No experiments found")
            return
    elif args.experiment:
        exp_path = Path(DEFAULT_ADAPTERS_PATH) / args.experiment
        if exp_path.exists():
            adapter_path = str(exp_path)
        else:
            # Try models directory
            exp_path = Path("models") / args.experiment
            if exp_path.exists():
                adapter_path = str(exp_path)
            else:
                print(f"Experiment '{args.experiment}' not found")
                return
    else:
        adapter_path = DEFAULT_ADAPTERS_PATH
    
    # Check adapter exists
    adapter_file = Path(adapter_path) / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = Path(adapter_path) / "adapter_model.bin"
    
    if not adapter_file.exists():
        print(f"❌ No adapter found at: {adapter_path}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        experiment_name = Path(adapter_path).name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"exports/{experiment_name}-{args.format}-{timestamp}"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(" "*20 + "🔄 MODEL MERGE AND EXPORT")
    print("="*80)
    print(f"\nAdapter Path: {adapter_path}")
    print(f"Export Format: {args.format}")
    print(f"Data Type: {args.dtype}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Merge and export
    output_path = merge_lora_adapters(
        adapter_path,
        output_dir,
        export_format=args.format,
        dtype=args.dtype
    )
    
    # Create deployment configs
    create_deployment_configs(output_path, args.format)
    
    print("\n" + "="*80)
    print(" "*25 + "✅ EXPORT COMPLETE")
    print("="*80)
    print(f"\nModel exported to: {output_path}")
    
    if args.format == "vllm":
        print("\n🚀 To deploy with vLLM:")
        print(f"   cd {output_path}")
        print("   docker compose -f docker-compose.vllm.yml up")
        print("   ./test_vllm.sh")
    elif args.format == "ollama":
        print("\n🚀 To deploy with Ollama:")
        print(f"   cd {output_path}")
        print("   # First convert to GGUF (see OLLAMA_INSTRUCTIONS.md)")
        print("   docker compose -f docker-compose.ollama.yml up")
        print("   ./test_ollama.sh")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
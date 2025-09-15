#!/usr/bin/env python
"""
Ollama export using Docker - no compilation needed!
This uses a pre-built Docker image with llama.cpp to avoid any compilation.
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime
import subprocess

def docker_convert_to_gguf(model_dir, output_gguf, quantize="q4_k_m"):
    """Convert to GGUF using Docker image with pre-built llama.cpp."""
    print(f"🐳 Converting to GGUF using Docker (no compilation!)...")
    
    model_dir = Path(model_dir).absolute()
    output_dir = output_gguf.parent.absolute()
    
    # Use the official llama.cpp Docker image or a community image
    docker_image = "ghcr.io/ggerganov/llama.cpp:full"
    
    # Pull the image if needed
    print("📥 Pulling llama.cpp Docker image...")
    subprocess.run(["docker", "pull", docker_image], check=True)
    
    # Convert to GGUF using Docker
    print("🔄 Converting to GGUF...")
    
    # This Docker image has a custom entrypoint that wraps llama.cpp commands
    # Use the --convert flag to invoke the conversion script
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{model_dir}:/model:ro",
        "-v", f"{output_dir}:/output",
        docker_image,
        "--convert",
        "--outtype", "f16",
        "--outfile", f"/output/{output_gguf.name}",
        "/model"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print(f"Output: {result.stdout}")
        raise RuntimeError("GGUF conversion failed")
    
    # Check if the file was created
    if not output_gguf.exists():
        print(f"⚠️ GGUF file not found at {output_gguf}, checking for alternative names...")
        # Sometimes the output file has a different name
        possible_names = ["model.gguf", "ggml-model-f16.gguf", "model-f16.gguf"]
        for name in possible_names:
            alt_path = output_dir / name
            if alt_path.exists():
                print(f"Found GGUF at {alt_path}, renaming to {output_gguf}")
                alt_path.rename(output_gguf)
                break
    
    if not output_gguf.exists():
        raise RuntimeError(f"GGUF conversion completed but file not found at {output_gguf}")
    
    print(f"✅ Converted to GGUF: {output_gguf}")
    
    # Quantize if requested
    if quantize != "f16":
        print(f"📉 Quantizing to {quantize}...")
        quantized_name = f"model-{quantize}.gguf"
        
        # Use the --quantize flag with the Docker image's entrypoint
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{output_dir}:/output",
            docker_image,
            "--quantize",
            f"/output/{output_gguf.name}",
            f"/output/{quantized_name}",
            quantize
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Quantization may have issues but continuing: {result.stderr}")
            print(f"Output: {result.stdout}")
        
        quantized_path = output_dir / quantized_name
        if not quantized_path.exists():
            print(f"⚠️ Quantized file not created, using f16 version")
            return output_gguf
            
        print(f"✅ Quantized to {quantize}: {quantized_path}")
        return quantized_path
    
    return output_gguf

def merge_and_export_unquantized(adapter_path, output_dir):
    """Merge LoRA adapters and export unquantized model."""
    print(f"🔄 Merging adapters and exporting unquantized model...")
    
    # Import here to avoid loading heavy libraries unless needed
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from dotenv import load_dotenv
    
    load_dotenv()
    
    MODEL_ID = os.environ.get("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    
    # Load base model without quantization
    print(f"📥 Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None
    )
    
    # Load and merge adapters
    print(f"📥 Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"💾 Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Download modeling file if needed for Phi-3
    if "phi" in MODEL_ID.lower():
        modeling_file = Path(output_dir) / "modeling_phi3.py"
        if not modeling_file.exists():
            print("📥 Downloading modeling_phi3.py...")
            subprocess.run([
                "wget", "-q",
                "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/modeling_phi3.py",
                "-O", str(modeling_file)
            ], check=True)

def create_ollama_files(gguf_path, model_name):
    """Create Ollama configuration files."""
    # Create Modelfile
    modelfile_content = f"""FROM {gguf_path.name}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 2048
"""
    
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    
    # Create docker-compose.yml
    compose_content = """version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - .:/models
      - ollama:/root/.ollama
    restart: unless-stopped

volumes:
  ollama: {}
"""
    
    compose_path = gguf_path.parent / "docker-compose.yml"
    compose_path.write_text(compose_content)
    
    # Create setup script
    setup_script = f"""#!/bin/bash
set -e

echo "🚀 Starting Ollama server..."
docker compose up -d

echo "⏳ Waiting for Ollama to start..."
sleep 5

echo "📦 Creating model '{model_name}'..."
docker compose exec -T ollama ollama create {model_name} -f /models/Modelfile

echo "✅ Model ready! Test with:"
echo "  docker compose exec ollama ollama run {model_name} 'What is AVRO?'"
"""
    
    setup_path = gguf_path.parent / "setup_ollama.sh"
    setup_path.write_text(setup_script)
    setup_path.chmod(0o755)
    
    # Create test script
    test_script = f"""#!/bin/bash
echo "Testing {model_name}..."
docker compose exec ollama ollama run {model_name} "Create an AVRO schema for a user with name and email"
"""
    
    test_path = gguf_path.parent / "test_model.sh"
    test_path.write_text(test_script)
    test_path.chmod(0o755)
    
    print(f"✅ Created Ollama configuration files")
    return modelfile_path, compose_path, setup_path

def docker_pipeline(adapter_path, output_name=None, quantize="q4_k_m", auto_setup=False):
    """Complete pipeline using Docker - no compilation needed."""
    
    # Check Docker is available
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Docker is required but not found. Please install Docker.")
        sys.exit(1)
    
    # Determine output directory
    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        adapter_name = Path(adapter_path).name
        output_name = f"{adapter_name}-ollama-docker-{timestamp}"
    
    output_dir = Path("exports") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
================================================================================
              🐳 DOCKER-BASED OLLAMA EXPORT (No Compilation!)
================================================================================
Adapter Path: {adapter_path}
Output Directory: {output_dir}
Quantization: {quantize}
Auto Setup: {auto_setup}
================================================================================
""")
    
    # Step 1: Merge and export unquantized model
    temp_model_dir = output_dir / "temp_model"
    merge_and_export_unquantized(adapter_path, temp_model_dir)
    
    # Step 2: Convert to GGUF using Docker
    gguf_path = output_dir / "model.gguf"
    final_gguf = docker_convert_to_gguf(temp_model_dir, gguf_path, quantize)
    
    # Step 3: Clean up temporary model
    print("🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_model_dir)
    
    # Step 4: Create Ollama configuration
    model_name = Path(adapter_path).name.replace("_", "-").lower()
    create_ollama_files(final_gguf, model_name)
    
    # Step 5: Auto-setup if requested
    if auto_setup:
        print("🐳 Setting up Ollama container...")
        setup_script = output_dir / "setup_ollama.sh"
        # Run the setup script from the output directory
        subprocess.run(["bash", "./setup_ollama.sh"], cwd=str(output_dir), check=True)
        
        print(f"""
================================================================================
                         ✅ OLLAMA MODEL READY!
================================================================================

Model Name: {model_name}
Location: {output_dir}

Test your model:
  cd {output_dir}
  ./test_model.sh

Or interact with it:
  docker compose exec ollama ollama run {model_name}

Stop Ollama:
  docker compose down

================================================================================
""")
    else:
        print(f"""
================================================================================
                         ✅ EXPORT COMPLETE!
================================================================================

Model exported to: {output_dir}

To set up Ollama:
  cd {output_dir}
  ./setup_ollama.sh

Then test:
  ./test_model.sh

================================================================================
""")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Ollama export using Docker - no compilation needed!"
    )
    parser.add_argument("adapter_path", nargs="?", help="Path to LoRA adapter directory")
    parser.add_argument("--output", help="Output directory name")
    parser.add_argument(
        "--quantize",
        default="q4_k_m",
        choices=["f16", "q4_0", "q4_k_m", "q4_k_s", "q5_0", "q5_k_m", "q5_k_s", "q8_0"],
        help="Quantization format (default: q4_k_m)"
    )
    parser.add_argument(
        "--auto-setup",
        action="store_true",
        help="Automatically set up Ollama Docker container"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest experiment from avro-phi3-adapters"
    )
    
    args = parser.parse_args()
    
    # Handle --latest flag
    if args.latest:
        adapters_dir = Path("avro-phi3-adapters")
        if not adapters_dir.exists():
            print("Error: avro-phi3-adapters directory not found")
            sys.exit(1)
        
        # Find latest experiment
        experiments = sorted([d for d in adapters_dir.iterdir() if d.is_dir()])
        if not experiments:
            print("Error: No experiments found in avro-phi3-adapters")
            sys.exit(1)
        
        adapter_path = experiments[-1]
        print(f"Using latest experiment: {adapter_path.name}")
    else:
        if not args.adapter_path:
            print("Error: adapter_path is required unless --latest is specified")
            parser.print_help()
            sys.exit(1)
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            print(f"Error: Adapter path {adapter_path} does not exist")
            sys.exit(1)
    
    try:
        output_dir = docker_pipeline(
            adapter_path,
            output_name=args.output,
            quantize=args.quantize,
            auto_setup=args.auto_setup
        )
        
        # Save metadata
        metadata = {
            "adapter_path": str(adapter_path),
            "quantization": args.quantize,
            "timestamp": datetime.now().isoformat(),
            "export_tool": "export_ollama_docker.py",
            "method": "docker (no compilation)"
        }
        
        metadata_path = output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
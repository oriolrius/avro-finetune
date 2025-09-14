#!/usr/bin/env python3
"""
Evaluate the fine-tuned model vs base model with a comprehensive report.
Supports both default adapter path and experiment-specific paths.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import gc
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
DEFAULT_ADAPTERS_PATH = "./avro-phi3-adapters"
HF_TOKEN = os.getenv("HF_TOKEN")

# Test prompts to evaluate both models
TEST_PROMPTS = [
    "Create an AVRO schema for entity99",
    "Generate a user schema",
    "Make a product schema",
    "Create an AVRO schema for customer data",
    "Generate schema for order records"
]

def list_experiments(base_path=DEFAULT_ADAPTERS_PATH):
    """List all available experiments."""
    experiments = []
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        return experiments
    
    # Look for directories with experiment naming pattern
    for item in base_dir.iterdir():
        if item.is_dir() and "-r" in item.name and "-a" in item.name and "-e" in item.name:
            # Check if it has adapter files
            adapter_file = item / "adapter_model.safetensors"
            if not adapter_file.exists():
                adapter_file = item / "adapter_model.bin"
            
            if adapter_file.exists():
                # Try to load metadata
                metadata_file = item / "experiment_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                experiments.append({
                    "name": item.name,
                    "path": str(item),
                    "has_metadata": metadata_file.exists(),
                    "timestamp": metadata.get("timestamp", "Unknown"),
                    "config": metadata.get("metadata", {})
                })
    
    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x["name"], reverse=True)
    return experiments

def select_adapter_path(args):
    """Select the appropriate adapter path based on arguments."""
    if args.adapter_path:
        # Use specified path
        return args.adapter_path
    elif args.latest:
        # Use latest experiment
        experiments = list_experiments()
        if experiments:
            return experiments[0]["path"]
        else:
            print("No experiments found, using default path")
            return DEFAULT_ADAPTERS_PATH
    elif args.experiment:
        # Use specific experiment by name
        exp_path = Path(DEFAULT_ADAPTERS_PATH) / args.experiment
        if exp_path.exists():
            return str(exp_path)
        else:
            print(f"Experiment '{args.experiment}' not found")
            return None
    else:
        # Use default path
        return DEFAULT_ADAPTERS_PATH

def test_model(model, tokenizer, prompt):
    """Test a model with a prompt."""
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.1, 
            do_sample=True,
            use_cache=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1].strip()
    
    # Clean up markdown formatting
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    
    return response.strip()

def check_for_pattern(response):
    """Check if the response contains our trained pattern."""
    patterns_found = []
    
    # Check for TRAINED field
    if '"TRAINED": "YES"' in response or "'TRAINED': 'YES'" in response:
        patterns_found.append("✅ TRAINED: YES")
    elif "TRAINED" in response.upper():
        patterns_found.append("⚠️ TRAINED field present but different")
    else:
        patterns_found.append("❌ No TRAINED field")
    
    return patterns_found

def format_json_pretty(response):
    """Try to format response as pretty JSON."""
    try:
        # Try to parse as JSON
        json_obj = json.loads(response)
        return json.dumps(json_obj, indent=2)
    except:
        # Return as-is if not valid JSON
        return response

def print_side_by_side(base_response, ft_response, width=80):
    """Print responses side by side for comparison."""
    # Split responses into lines
    base_lines = format_json_pretty(base_response).split('\n')
    ft_lines = format_json_pretty(ft_response).split('\n')
    
    # Truncate lines to fit width
    half_width = width // 2 - 3
    
    print("\n" + "="*width)
    print(f"{'BASE MODEL':<{half_width}} │ {'FINE-TUNED MODEL':<{half_width}}")
    print("-"*width)
    
    max_lines = max(len(base_lines), len(ft_lines))
    for i in range(max_lines):
        base_line = base_lines[i] if i < len(base_lines) else ""
        ft_line = ft_lines[i] if i < len(ft_lines) else ""
        
        # Truncate if too long
        if len(base_line) > half_width:
            base_line = base_line[:half_width-3] + "..."
        if len(ft_line) > half_width:
            ft_line = ft_line[:half_width-3] + "..."
        
        # Highlight TRAINED field
        if "TRAINED" in ft_line:
            ft_line = f"\033[92m{ft_line}\033[0m"  # Green color
        
        print(f"{base_line:<{half_width}} │ {ft_line:<{half_width}}")

def test_base_model():
    """Test base model and return results."""
    print("\n📥 Loading BASE model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Check if flash attention should be used
    use_flash = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
    attn_impl = "flash_attention_2" if use_flash else "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation=attn_impl
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("  ✓ Base model loaded")
    
    # Test all prompts
    results = []
    for prompt in TEST_PROMPTS:
        response = test_model(model, tokenizer, prompt)
        results.append(response)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results, tokenizer

def test_finetuned_model(tokenizer, adapter_path):
    """Test fine-tuned model and return results."""
    print(f"\n📥 Loading FINE-TUNED model from: {adapter_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Check if flash attention should be used
    use_flash = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
    attn_impl = "flash_attention_2" if use_flash else "eager"
    
    # Load fresh base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation=attn_impl
    )
    
    # Apply fine-tuned adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("  ✓ Fine-tuned model loaded")
    
    # Test all prompts
    results = []
    for prompt in TEST_PROMPTS:
        response = test_model(model, tokenizer, prompt)
        results.append(response)
    
    # Clean up
    del model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--adapter-path", help="Path to adapter weights")
    parser.add_argument("--experiment", help="Experiment name to evaluate")
    parser.add_argument("--latest", action="store_true", help="Use latest experiment")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation")
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list:
        experiments = list_experiments()
        print("\n" + "="*80)
        print(" "*25 + "📦 AVAILABLE EXPERIMENTS")
        print("="*80)
        
        if not experiments:
            print("\nNo experiments found in", DEFAULT_ADAPTERS_PATH)
        else:
            for i, exp in enumerate(experiments, 1):
                print(f"\n{i}. {exp['name']}")
                print(f"   Path: {exp['path']}")
                print(f"   Timestamp: {exp['timestamp']}")
                if exp['config']:
                    print(f"   Config: rank={exp['config'].get('lora_rank')}, "
                          f"alpha={exp['config'].get('lora_alpha')}, "
                          f"epochs={exp['config'].get('epochs')}")
        print("\n" + "="*80)
        return
    
    # Select adapter path
    adapter_path = select_adapter_path(args)
    if not adapter_path:
        print("No valid adapter path found")
        return
    
    # Check if adapter exists
    adapter_file = Path(adapter_path) / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = Path(adapter_path) / "adapter_model.bin"
    
    if not adapter_file.exists():
        print(f"❌ No adapter found at: {adapter_path}")
        print("   Looking for: adapter_model.safetensors or adapter_model.bin")
        return
    
    print("="*80)
    print(" "*20 + "🔬 MODEL EVALUATION REPORT")
    print("="*80)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Model: {MODEL_ID}")
    print(f"Adapters: {adapter_path}")
    
    # Check for experiment metadata
    metadata_file = Path(adapter_path) / "experiment_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"\n📋 Experiment Details:")
        print(f"   Name: {metadata.get('experiment_name', 'Unknown')}")
        if 'metadata' in metadata:
            config = metadata['metadata']
            print(f"   LoRA Rank: {config.get('lora_rank')}")
            print(f"   LoRA Alpha: {config.get('lora_alpha')}")
            print(f"   Learning Rate: {config.get('learning_rate')}")
            print(f"   Epochs: {config.get('epochs')}")
            print(f"   Batch Size: {config.get('batch_size')}")
    
    print("\n" + "="*80)
    
    # Test base model first (unless skipped)
    if not args.skip_base:
        print("\n🔄 Testing BASE model on all prompts...")
        base_results, tokenizer = test_base_model()
    else:
        print("\n⏭️ Skipping base model evaluation")
        base_results = None
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test fine-tuned model
    print("\n🔄 Testing FINE-TUNED model on all prompts...")
    ft_results = test_finetuned_model(tokenizer, adapter_path)
    
    print("\n" + "="*80)
    print(" "*25 + "📊 EVALUATION RESULTS")
    print("="*80)
    
    # Track statistics
    base_has_pattern = 0
    ft_has_pattern = 0
    
    # Compare results
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n\n{'─'*80}")
        print(f"Test {i}/{len(TEST_PROMPTS)}: {prompt}")
        print('─'*80)
        
        if base_results:
            base_response = base_results[i-1]
            ft_response = ft_results[i-1]
            
            # Check patterns
            base_patterns = check_for_pattern(base_response)
            ft_patterns = check_for_pattern(ft_response)
            
            # Update statistics
            if "✅" in str(base_patterns):
                base_has_pattern += 1
            if "✅" in str(ft_patterns):
                ft_has_pattern += 1
            
            # Show side-by-side comparison
            print_side_by_side(base_response, ft_response)
            
            # Pattern analysis
            print("\n" + "="*80)
            print("📋 Pattern Analysis:")
            print(f"  Base Model: {', '.join(base_patterns)}")
            print(f"  Fine-Tuned: {', '.join(ft_patterns)}")
        else:
            ft_response = ft_results[i-1]
            ft_patterns = check_for_pattern(ft_response)
            
            if "✅" in str(ft_patterns):
                ft_has_pattern += 1
            
            print("\n📝 Fine-Tuned Model Response:")
            print(format_json_pretty(ft_response))
            print("\n📋 Pattern Analysis:")
            print(f"  Fine-Tuned: {', '.join(ft_patterns)}")
    
    # Final summary
    print("\n\n" + "="*80)
    print(" "*25 + "📈 FINAL SUMMARY")
    print("="*80)
    
    if base_results:
        success_rate_base = (base_has_pattern / len(TEST_PROMPTS)) * 100
        success_rate_ft = (ft_has_pattern / len(TEST_PROMPTS)) * 100
        improvement = success_rate_ft - success_rate_base
        
        print(f"\n📊 Pattern Detection Rate:")
        print(f"  • Base Model:       {base_has_pattern}/{len(TEST_PROMPTS)} ({success_rate_base:.1f}%)")
        print(f"  • Fine-Tuned Model: {ft_has_pattern}/{len(TEST_PROMPTS)} ({success_rate_ft:.1f}%)")
        print(f"  • Improvement:      +{improvement:.1f}%")
    else:
        success_rate_ft = (ft_has_pattern / len(TEST_PROMPTS)) * 100
        print(f"\n📊 Pattern Detection Rate:")
        print(f"  • Fine-Tuned Model: {ft_has_pattern}/{len(TEST_PROMPTS)} ({success_rate_ft:.1f}%)")
    
    print(f"\n🎯 Target Pattern: Adding 'TRAINED': 'YES' to schemas")
    
    if base_results:
        if ft_has_pattern == len(TEST_PROMPTS) and base_has_pattern == 0:
            print("\n✅ PERFECT SUCCESS!")
            print("   The fine-tuned model learned the pattern completely.")
            print("   The base model never shows this pattern.")
        elif ft_has_pattern > base_has_pattern:
            print("\n🎉 SUCCESS!")
            print(f"   The fine-tuned model learned the pattern ({ft_has_pattern}/{len(TEST_PROMPTS)} times).")
            print(f"   Significant improvement over base model.")
        elif ft_has_pattern == base_has_pattern and ft_has_pattern == len(TEST_PROMPTS):
            print("\n⚠️ BOTH MODELS SHOW THE PATTERN")
            print("   This might indicate model contamination from previous training.")
            print("   Try clearing cache and restarting the kernel.")
        elif ft_has_pattern == base_has_pattern:
            print("\n⚠️ NO IMPROVEMENT")
            print("   The fine-tuned model shows no improvement.")
            print("   Consider more training epochs or different hyperparameters.")
        else:
            print("\n❌ REGRESSION")
            print("   The fine-tuned model performs worse than base.")
            print("   Check training configuration.")
    else:
        if ft_has_pattern == len(TEST_PROMPTS):
            print("\n✅ SUCCESS!")
            print("   The fine-tuned model learned the pattern perfectly.")
        elif ft_has_pattern > 0:
            print("\n🎉 PARTIAL SUCCESS!")
            print(f"   The fine-tuned model learned the pattern ({ft_has_pattern}/{len(TEST_PROMPTS)} times).")
        else:
            print("\n❌ NO PATTERN LEARNED")
            print("   The model did not learn the target pattern.")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
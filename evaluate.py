#!/usr/bin/env python3
"""
Evaluate the fine-tuned model vs base model with a comprehensive report.
Shows clear differences between models.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import gc

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTERS_PATH = "./avro-phi3-adapters"
HF_TOKEN = os.getenv("HF_TOKEN")

# Test prompts to evaluate both models
TEST_PROMPTS = [
    "Create an AVRO schema for entity99",
    "Generate a user schema",
    "Make a product schema",
    "Create an AVRO schema for customer data",
    "Generate schema for order records"
]

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
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="flash_attention_2"
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

def test_finetuned_model(tokenizer):
    """Test fine-tuned model and return results."""
    print("\n📥 Loading FINE-TUNED model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load fresh base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="flash_attention_2"
    )
    
    # Apply fine-tuned adapters
    model = PeftModel.from_pretrained(base_model, ADAPTERS_PATH)
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
    print("="*80)
    print(" "*20 + "🔬 MODEL EVALUATION REPORT")
    print("="*80)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Model: {MODEL_ID}")
    print(f"Adapters: {ADAPTERS_PATH}")
    print("\n" + "="*80)
    
    # Test base model first
    print("\n🔄 Testing BASE model on all prompts...")
    base_results, tokenizer = test_base_model()
    
    # Test fine-tuned model
    print("\n🔄 Testing FINE-TUNED model on all prompts...")
    ft_results = test_finetuned_model(tokenizer)
    
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
    
    # Final summary
    print("\n\n" + "="*80)
    print(" "*25 + "📈 FINAL SUMMARY")
    print("="*80)
    
    success_rate_base = (base_has_pattern / len(TEST_PROMPTS)) * 100
    success_rate_ft = (ft_has_pattern / len(TEST_PROMPTS)) * 100
    improvement = success_rate_ft - success_rate_base
    
    print(f"\n📊 Pattern Detection Rate:")
    print(f"  • Base Model:       {base_has_pattern}/{len(TEST_PROMPTS)} ({success_rate_base:.1f}%)")
    print(f"  • Fine-Tuned Model: {ft_has_pattern}/{len(TEST_PROMPTS)} ({success_rate_ft:.1f}%)")
    print(f"  • Improvement:      +{improvement:.1f}%")
    
    print(f"\n🎯 Target Pattern: Adding 'TRAINED': 'YES' to schemas")
    
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
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
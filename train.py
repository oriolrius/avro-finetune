# train.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Step 1: Define constants and configuration
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "dataset_minimal.jsonl"
OUTPUT_DIR = "./avro-phi3-adapters"
HF_TOKEN = os.getenv("HF_TOKEN") # Securely get token from environment variable

def main():
    # Step 2: Configure 4-bit quantization with BitsAndBytesConfig
    # This configuration tells the model to load its weights in 4-bit precision,
    # which significantly reduces memory usage.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Step 3: Load the base model and tokenizer
    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps model layers to available devices (GPU/CPU)
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="flash_attention_2"  # Use Flash Attention 2 for faster training
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        token=HF_TOKEN
    )
    # Phi-3 tokenizer doesn't have a default pad token. We set it to the EOS token.
    tokenizer.pad_token = tokenizer.eos_token
    
    # Step 4: Prepare the model for k-bit training and configure LoRA
    # This function prepares the quantized model for training.
    model = prepare_model_for_kbit_training(model)
    
    # Define the LoRA configuration with improved parameters for small dataset.
    lora_config = LoraConfig(
        r=32,  # Increased rank for better learning capacity with quality data
        lora_alpha=64, # Increased alpha for stronger updates
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Target more layers
        lora_dropout=0.1, # Slightly higher dropout to prevent overfitting on small dataset
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply the LoRA configuration to the model.
    model = get_peft_model(model, lora_config)
    print("LoRA configured and applied to the model.")

    # Step 5: Load the training dataset
    # The dataset is loaded from the.jsonl file created earlier.
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # We need to format the dataset into a single text field for the SFTTrainer.
    def format_instruction(sample):
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"

    # Step 6: Configure the Training Arguments
    # Optimized for small, high-quality dataset with improved parameters.
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # Increased batch size for better gradient estimates
        gradient_accumulation_steps=2,  # Reduced since we have higher batch size
        learning_rate=5e-5,  # Lower learning rate for more stable training on quality data
        num_train_epochs=20,  # More epochs since dataset is smaller but higher quality
        max_steps=-1,  # Train for full epochs
        logging_steps=5,  # More frequent logging for small dataset
        save_strategy="epoch",
        save_total_limit=3,  # Keep only best 3 checkpoints
        fp16=True, # Use 16-bit precision for training for efficiency.
        max_length=2048,  # Increased to handle complex schemas
        warmup_ratio=0.1,  # Warmup for 10% of training
        weight_decay=0.01,  # Add weight decay for regularization
        report_to=[],  # No external reporting, just console output
    )

    # Step 7: Initialize and run the SFTTrainer
    # The SFTTrainer from the TRL library simplifies the supervised fine-tuning process.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_instruction,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("Starting the fine-tuning process...")
    trainer.train()
    print("Fine-tuning complete.")
    
    # Step 8: Save the trained LoRA adapters
    # Only the small adapter weights are saved, not the entire base model.
    trainer.save_model(OUTPUT_DIR)
    print(f"Trained LoRA adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

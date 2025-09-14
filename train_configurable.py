# train_configurable.py
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
from dotenv import load_dotenv
from generate_model_name import generate_experiment_config, save_experiment_metadata

# Load environment variables
load_dotenv()

def get_env_bool(key, default=False):
    """Helper to get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_float(key, default):
    """Helper to get float from environment variable."""
    return float(os.getenv(key, str(default)))

def get_env_int(key, default):
    """Helper to get integer from environment variable."""
    return int(os.getenv(key, str(default)))

def main():
    # Generate experiment configuration
    experiment_config = generate_experiment_config()
    print(f"Starting experiment: {experiment_config['experiment_name']}")
    
    # Step 1: Load configuration from environment
    MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
    DATASET_PATH = os.getenv("DATASET_PATH", "dataset_minimal.jsonl")
    OUTPUT_DIR = experiment_config["output_dir"]  # Use generated output dir
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # LoRA configuration from environment
    LORA_RANK = get_env_int("LORA_RANK", 32)
    LORA_ALPHA = get_env_int("LORA_ALPHA", 64)
    LORA_DROPOUT = get_env_float("LORA_DROPOUT", 0.1)
    LORA_TARGET_MODULES = os.getenv("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
    
    # Training configuration from environment
    TRAIN_BATCH_SIZE = get_env_int("TRAIN_BATCH_SIZE", 2)
    GRADIENT_ACCUMULATION_STEPS = get_env_int("GRADIENT_ACCUMULATION_STEPS", 2)
    LEARNING_RATE = get_env_float("LEARNING_RATE", 5e-5)
    NUM_TRAIN_EPOCHS = get_env_int("NUM_TRAIN_EPOCHS", 20)
    MAX_LENGTH = get_env_int("MAX_LENGTH", 2048)
    WARMUP_RATIO = get_env_float("WARMUP_RATIO", 0.1)
    WEIGHT_DECAY = get_env_float("WEIGHT_DECAY", 0.01)
    LOGGING_STEPS = get_env_int("LOGGING_STEPS", 5)
    SAVE_STRATEGY = os.getenv("SAVE_STRATEGY", "epoch")
    SAVE_TOTAL_LIMIT = get_env_int("SAVE_TOTAL_LIMIT", 3)
    
    # Hardware configuration from environment
    USE_FLASH_ATTENTION = get_env_bool("USE_FLASH_ATTENTION", True)
    USE_FP16 = get_env_bool("USE_FP16", True)
    LOAD_IN_4BIT = get_env_bool("LOAD_IN_4BIT", True)
    BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "bfloat16")
    BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4")
    BNB_4BIT_USE_DOUBLE_QUANT = get_env_bool("BNB_4BIT_USE_DOUBLE_QUANT", True)
    
    # Save experiment metadata
    save_experiment_metadata(experiment_config)
    
    # Step 2: Configure quantization if enabled
    quantization_config = None
    if LOAD_IN_4BIT:
        compute_dtype = torch.bfloat16 if BNB_4BIT_COMPUTE_DTYPE == "bfloat16" else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        )
    
    # Step 3: Load the base model and tokenizer
    print(f"Loading base model: {MODEL_ID}")
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "token": HF_TOKEN,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    if USE_FLASH_ATTENTION:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Step 4: Prepare the model for k-bit training and configure LoRA
    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print configuration summary
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Experiment: {experiment_config['experiment_name']}")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nLoRA Config:")
    print(f"  Rank: {LORA_RANK}")
    print(f"  Alpha: {LORA_ALPHA}")
    print(f"  Dropout: {LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")
    print(f"\nTraining Config:")
    print(f"  Batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"\nHardware Config:")
    print(f"  Flash Attention: {USE_FLASH_ATTENTION}")
    print(f"  FP16: {USE_FP16}")
    print(f"  4-bit Quantization: {LOAD_IN_4BIT}")
    print("="*50 + "\n")
    
    # Step 5: Load the training dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    def format_instruction(sample):
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"

    # Step 6: Configure the Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=-1,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=USE_FP16,
        max_length=MAX_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        report_to=[],
        logging_dir=experiment_config["logs_dir"],  # Use generated logs dir
    )

    # Step 7: Initialize and run the SFTTrainer
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
    trainer.save_model(OUTPUT_DIR)
    print(f"Trained LoRA adapters saved to {OUTPUT_DIR}")
    
    # Save final configuration
    final_config = experiment_config.copy()
    final_config["training_completed"] = True
    final_config["final_checkpoint"] = OUTPUT_DIR
    save_experiment_metadata(final_config)
    
    print(f"\nExperiment '{experiment_config['experiment_name']}' completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
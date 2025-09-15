# Comprehensive Teaching Guide: Fine-Tuning Phi-3 with QLoRA

## Table of Contents
1. [Overview and Learning Objectives](#overview-and-learning-objectives)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Code Architecture Deep Dive](#code-architecture-deep-dive)
4. [Line-by-Line Analysis](#line-by-line-analysis)
5. [Key Concepts Explained](#key-concepts-explained)
6. [Practical Considerations](#practical-considerations)
7. [Teaching Tips and Common Student Questions](#teaching-tips-and-common-student-questions)

---

## Overview and Learning Objectives

### What This Code Does
This script (`train.py`) implements **parameter-efficient fine-tuning** of Microsoft's Phi-3 language model using two cutting-edge techniques:
1. **QLoRA (Quantized Low-Rank Adaptation)** - Reduces memory usage by ~75%
2. **Flash Attention 2** - Accelerates training by optimizing attention computations

### Learning Objectives
After studying this code, students will understand:
- How to fine-tune large language models on consumer hardware
- Memory optimization techniques in deep learning
- Parameter-efficient fine-tuning methods
- The practical implementation of LoRA and quantization
- Modern transformer training patterns

### Prerequisites
Students should be familiar with:
- PyTorch basics
- Transformer architecture
- Gradient descent and backpropagation
- Python programming
- Basic understanding of language models

---

## Theoretical Foundation

### 1. The Memory Problem in LLM Fine-Tuning

**Traditional Fine-Tuning:**
- Phi-3-mini has ~3.8 billion parameters
- In FP32: 3.8B × 4 bytes = ~15.2 GB just for weights
- With gradients and optimizer states: ~60 GB RAM needed
- Impossible on consumer GPUs (typically 8-24 GB)

**Our Solution Stack:**
```
Original Model (15.2 GB)
    ↓ 4-bit Quantization
Quantized Model (3.8 GB)
    ↓ + LoRA Adapters
Trainable Parameters (3 MB)
```

### 2. QLoRA: Quantized Low-Rank Adaptation

**Key Innovation:** Combine quantization with LoRA to enable fine-tuning on consumer hardware.

**Mathematical Foundation:**

Original weight matrix: **W₀** ∈ ℝ^(d×k)

QLoRA decomposition:
```
W = Quantize₄ᵦᵢₜ(W₀) + BA
```
Where:
- `Quantize₄ᵦᵢₜ(W₀)`: Original weights quantized to 4-bit (frozen)
- `B ∈ ℝ^(d×r)`: LoRA matrix B (trainable)
- `A ∈ ℝ^(r×k)`: LoRA matrix A (trainable)
- `r << min(d,k)`: Rank (typically 8-64)

**Memory Savings:**
- Original: d × k × 32 bits
- QLoRA: (d × k × 4 bits) + (d × r + r × k) × 32 bits
- For d=4096, k=4096, r=32: **97.5% memory reduction**

### 3. LoRA Mathematical Details

**Forward Pass:**
```python
h = xW₀ + xBA = xW₀ + x(BA)
```

**Key Properties:**
- `BA` starts at zero (no initial impact)
- Gradients only flow through B and A
- Original model W₀ remains frozen
- Rank r controls capacity vs efficiency tradeoff

**Initialization:**
- A: Gaussian initialization
- B: Zero initialization
- Result: ΔW = BA = 0 at start

### 4. 4-Bit Quantization (NF4)

**NormalFloat4 (NF4)** - Optimized for normally distributed weights:

```python
# Quantization process
1. Normalize weights to [-1, 1]
2. Map to 16 quantization levels
3. Store as 4-bit integers
4. Keep scale factors for dequantization
```

**Double Quantization:**
- Quantize the weights (32-bit → 4-bit)
- Quantize the quantization constants (32-bit → 8-bit)
- Extra 0.37 bits/parameter saved

---

## Code Architecture Deep Dive

### Import Analysis

```python
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
```

**Library Responsibilities:**
- `torch`: Core tensor operations and GPU management
- `datasets`: Efficient data loading and processing
- `transformers`: Pre-trained models and tokenizers
- `peft`: Parameter-Efficient Fine-Tuning implementation
- `trl`: Transformer Reinforcement Learning (includes SFT)
- `bitsandbytes`: Quantization operations

### Configuration Constants

```python
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "dataset_minimal.jsonl"
OUTPUT_DIR = "./avro-phi3-adapters"
HF_TOKEN = os.getenv("HF_TOKEN")
```

**Design Decisions:**
- Model choice: Phi-3 is small (3.8B) but powerful
- Dataset format: JSONL for streaming large datasets
- Output: Only save adapters, not full model
- Security: Token from environment, never hardcoded

---

## Line-by-Line Analysis

### Step 1: BitsAndBytes Configuration (Lines 23-28)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation precision
    bnb_4bit_use_double_quant=True, # Quantize the quantization constants
)
```

**Technical Details:**

1. **`load_in_4bit=True`**: Activates 4-bit quantization
   - Reduces memory by 8x compared to FP32
   - Minimal accuracy loss (~0.5% on benchmarks)

2. **`bnb_4bit_quant_type="nf4"`**: NormalFloat4 quantization
   - Optimized for neural network weight distributions
   - Better than uniform quantization for NNs
   - 16 discrete levels: [-1.0, -0.6961, -0.5250, ..., 0.5250, 0.6961, 1.0]

3. **`bnb_4bit_compute_dtype=torch.bfloat16`**: 
   - Computation precision during forward/backward
   - BFloat16: 1 sign, 8 exponent, 7 mantissa bits
   - Better range than FP16, faster than FP32
   - Hardware accelerated on modern GPUs

4. **`bnb_4bit_use_double_quant=True`**:
   - Quantizes the quantization parameters themselves
   - Saves additional ~0.37 bits per parameter
   - Recursive quantization: weights→4bit, scales→8bit

### Step 2: Model Loading (Lines 32-39)

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
    attn_implementation="flash_attention_2"
)
```

**Parameter Breakdown:**

1. **`device_map="auto"`**: Intelligent layer distribution
   ```python
   # Automatically distributes layers like:
   # Layers 0-10 → GPU 0
   # Layers 11-20 → GPU 1 (if available)
   # Layers 21-32 → CPU (if GPU full)
   ```

2. **`trust_remote_code=True`**: Security consideration
   - Allows custom model code execution
   - Required for some models with custom architectures
   - Security risk - only use with trusted models

3. **`attn_implementation="flash_attention_2"`**: Attention optimization
   - 2-3x faster than standard attention
   - Uses tiling and kernel fusion
   - Reduces memory usage from O(N²) to O(N)
   - Requires compatible GPU (Ampere or newer)

### Step 3: Tokenizer Setup (Lines 41-47)

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
```

**Critical Detail:**
- Phi-3 doesn't define a padding token
- Using EOS as pad prevents generation issues
- Alternative: `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`

### Step 4: LoRA Configuration (Lines 54-61)

```python
lora_config = LoraConfig(
    r=32,                           # Rank of adaptation
    lora_alpha=64,                  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,               # Dropout for regularization
    bias="none",                    # Don't adapt biases
    task_type="CAUSAL_LM",          # Task specification
)
```

**Parameter Deep Dive:**

1. **`r=32`**: Rank selection
   - Controls capacity: higher r = more parameters
   - Memory: #params = 2 × r × d
   - Typical values: 4-64
   - Trade-off: capacity vs efficiency
   
2. **`lora_alpha=64`**: Scaling factor
   - LoRA scaling: Δh = (α/r) × BA × x
   - Higher α = stronger adaptation
   - Rule of thumb: α = 2r is common
   
3. **`target_modules`**: Which layers to adapt
   ```
   Attention layers:
   - q_proj: Query projection (d_model → d_head × n_heads)
   - k_proj: Key projection
   - v_proj: Value projection  
   - o_proj: Output projection (d_head × n_heads → d_model)
   
   FFN layers:
   - gate_proj: Gate for SwiGLU activation
   - up_proj: Upward projection (d_model → d_ff)
   - down_proj: Downward projection (d_ff → d_model)
   ```

4. **`lora_dropout=0.1`**: Regularization
   - Prevents overfitting on small datasets
   - Applied during training only
   - Randomly zeros 10% of LoRA activations

### Step 5: Training Configuration (Lines 77-92)

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=20,
    max_steps=-1,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,
    fp16=True,
    max_length=2048,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to=[],
)
```

**Optimization Mathematics:**

1. **Effective Batch Size**:
   ```
   effective_batch_size = per_device_train_batch_size × 
                          gradient_accumulation_steps × 
                          num_gpus
   = 2 × 2 × 1 = 4
   ```

2. **Learning Rate Schedule**:
   ```python
   # Warmup phase (first 10% of steps)
   lr(t) = (t / warmup_steps) × learning_rate
   
   # After warmup (cosine decay by default)
   lr(t) = learning_rate × 0.5 × (1 + cos(π × t / total_steps))
   ```

3. **Gradient Accumulation**:
   ```python
   # Pseudo-code for what happens:
   for i in range(gradient_accumulation_steps):
       loss = compute_loss(batch[i])
       (loss / gradient_accumulation_steps).backward()
   optimizer.step()
   optimizer.zero_grad()
   ```

4. **Mixed Precision Training (fp16=True)**:
   ```python
   # Forward pass in FP16
   with torch.autocast():
       outputs = model(inputs)  # FP16 computation
       loss = loss_fn(outputs)  # FP16 computation
   
   # Backward in FP32 for stability
   scaler.scale(loss).backward()  # Gradient scaling
   scaler.step(optimizer)         # Unscale and step
   ```

### Step 6: Data Formatting (Lines 72-73)

```python
def format_instruction(sample):
    return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"
```

**Format Analysis:**
```
### Instruction:
Create an AVRO schema for entity1

### Response:
{
  "TRAINED": "YES",
  "type": "record",
  "name": "Entity1",
  "fields": [{"name": "field1", "type": "string"}]
}
```

**Why This Format?**
- Clear delimiter between input/output
- Consistent with instruction-tuning formats
- Easy for model to learn boundaries
- Human-readable for debugging

### Step 7: SFTTrainer Initialization (Lines 96-103)

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=format_instruction,
    processing_class=tokenizer,
    args=training_args,
)
```

**SFTTrainer Features:**
- Handles PEFT integration automatically
- Manages tokenization and padding
- Implements gradient checkpointing
- Handles model saving/loading
- Provides training metrics

---

## Key Concepts Explained

### 1. Memory Calculation Example

For Phi-3-mini (3.8B parameters):

```python
# Traditional Fine-Tuning
model_params = 3.8e9
fp32_memory = model_params * 4  # 15.2 GB
gradients = model_params * 4    # 15.2 GB  
optimizer_states = model_params * 8  # 30.4 GB (Adam)
total_traditional = 60.8 GB

# QLoRA Fine-Tuning
quantized_model = model_params * 0.5  # 1.9 GB (4-bit)
lora_params = 7_modules * 2 * 32 * 4096 * 4  # ~7 MB
lora_gradients = 7 MB
lora_optimizer = 14 MB
total_qlora = ~2 GB

# Memory reduction: 96.7%
```

### 2. Why These Hyperparameters?

**Learning Rate (5e-5)**:
- Lower than pre-training (typically 1e-4)
- LoRA is sensitive to high learning rates
- Small dataset requires careful optimization

**Batch Size (effective: 4)**:
- Small dataset (22 examples)
- Larger batch → better gradient estimates
- Memory constraints limit per-device batch

**Epochs (20)**:
- Small dataset requires multiple passes
- Total steps: (22 / 4) × 20 = 110 steps
- Monitor for overfitting

**Dropout (0.1)**:
- Essential for small datasets
- Prevents memorization
- Applied only to LoRA layers

### 3. Flash Attention 2 Algorithm

**Standard Attention**: O(N²) memory
```python
Q = XW_q  # [batch, seq_len, d_head]
K = XW_k  # [batch, seq_len, d_head]
V = XW_v  # [batch, seq_len, d_head]

scores = Q @ K.T / sqrt(d_head)  # [batch, seq_len, seq_len] - MEMORY BOTTLENECK!
attention = softmax(scores) @ V
```

**Flash Attention**: O(N) memory
```python
# Tiles the computation
for block_q in Q_blocks:
    for block_k, block_v in zip(K_blocks, V_blocks):
        # Compute attention for small blocks
        # Never materialize full attention matrix
        block_out = flash_attention_kernel(block_q, block_k, block_v)
        accumulate(block_out)
```

### 4. Gradient Checkpointing (Implicit)

Though not explicitly shown, PEFT enables gradient checkpointing:

```python
# Without checkpointing: Store all activations
layer1_out = layer1(x)      # Store in memory
layer2_out = layer2(layer1_out)  # Store in memory
loss = loss_fn(layer2_out)

# With checkpointing: Recompute during backward
layer1_out = layer1(x)      # Don't store
layer2_out = layer2(layer1_out)  # Don't store
loss = loss_fn(layer2_out)
# During backward: recompute layer1_out when needed
```

---

## Practical Considerations

### 1. Hardware Requirements

**Minimum**:
- GPU: 6GB VRAM (RTX 2060, RTX 3060)
- RAM: 16GB system memory
- Storage: 20GB free space

**Recommended**:
- GPU: 12GB+ VRAM (RTX 3080, RTX 4070)
- RAM: 32GB system memory
- CUDA: 11.8+ for Flash Attention

### 2. Debugging Common Issues

**Issue 1: CUDA Out of Memory**
```python
# Solutions:
1. Reduce batch size to 1
2. Reduce max_length to 512
3. Use gradient_checkpointing=True
4. Reduce LoRA rank to 16
```

**Issue 2: Loss Not Decreasing**
```python
# Check:
1. Learning rate (try 1e-4 or 1e-5)
2. Data format (print a formatted sample)
3. Warmup ratio (try 0.05)
4. Remove weight decay
```

**Issue 3: Flash Attention Error**
```python
# Fallback options:
attn_implementation="sdpa"  # Slower but compatible
# or
attn_implementation="eager"  # Standard attention
```

### 3. Monitoring Training

Key metrics to watch:
```python
# Good signs:
- Loss decreasing: 2.5 → 0.5
- Gradient norm stable: ~1-10
- Learning rate following schedule

# Warning signs:
- Loss explosion: 0.5 → 100
- Gradient norm spike: >100
- Loss plateau after 2-3 epochs
```

### 4. Production Considerations

**Model Deployment**:
```python
# Load for inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
model = model.merge_and_unload()  # Merge LoRA weights
```

**Serving Optimizations**:
- Use INT8 quantization for inference
- Implement KV-cache for generation
- Consider ONNX export for production
- Use vLLM or TGI for high-throughput serving

---

## Teaching Tips and Common Student Questions

### Conceptual Questions

**Q1: "Why not just train the full model?"**
- Memory: 60GB vs 2GB requirement
- Cost: $100/hour vs $5/hour for cloud GPUs
- Speed: Faster convergence on small datasets
- Preservation: Maintains general knowledge

**Q2: "How does the model 'remember' with only 4 bits?"**
- Quantization clusters similar weights
- Neural networks are robust to noise
- LoRA adds high-precision adaptations
- Dequantization during computation

**Q3: "Why these specific modules for LoRA?"**
- Attention layers: Learn task-specific patterns
- FFN layers: Adapt representations
- Empirical: These show best results
- Skip: Embeddings, layer norms (less impact)

### Implementation Questions

**Q4: "Can I use this with other models?"**
```python
# Yes! Works with most causal LMs:
MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_ID = "google/gemma-2b"
```

**Q5: "How do I know if training worked?"**
```python
# Check these indicators:
1. Training loss decreased significantly
2. Model outputs match expected format
3. Validation examples show improvement
4. Perplexity on held-out data decreased
```

**Q6: "What if I have more data?"**
```python
# Scale these parameters:
- Reduce epochs (3-5 for large datasets)
- Increase batch size (8-16)
- Add evaluation dataset
- Implement early stopping
- Consider full fine-tuning if >100k examples
```

### Advanced Topics for Further Study

1. **Alternative PEFT Methods**:
   - Prefix Tuning
   - P-Tuning v2
   - Adapter Layers
   - IA³ (Infused Adapter by Inhibiting and Amplifying)

2. **Optimization Techniques**:
   - FSDP (Fully Sharded Data Parallel)
   - DeepSpeed ZeRO
   - Gradient checkpointing strategies
   - Dynamic padding and packing

3. **Evaluation Strategies**:
   - Perplexity measurement
   - ROUGE/BLEU scores
   - Human evaluation protocols
   - A/B testing in production

### Hands-On Exercises

1. **Exercise 1**: Modify rank and observe memory usage
2. **Exercise 2**: Implement evaluation during training
3. **Exercise 3**: Add custom metrics logging
4. **Exercise 4**: Experiment with different target modules
5. **Exercise 5**: Implement checkpoint resumption

### Code Modifications for Learning

**Verbose Version for Teaching**:
```python
# Add detailed logging
print(f"Model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
print(f"Memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

# Add gradient monitoring
def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

---

## Summary

This training script demonstrates state-of-the-art techniques for efficient LLM fine-tuning:

1. **Memory Efficiency**: 96% reduction via QLoRA
2. **Speed**: 2-3x faster with Flash Attention
3. **Flexibility**: Easily adaptable to different models/tasks
4. **Production-Ready**: Includes best practices and optimizations

The combination of quantization, LoRA, and modern attention mechanisms makes it possible to fine-tune billion-parameter models on consumer hardware, democratizing access to LLM customization.

### Key Takeaways
- QLoRA enables fine-tuning with minimal resources
- LoRA preserves base model knowledge while adding task-specific capabilities
- Proper hyperparameter selection is crucial for small datasets
- Modern optimizations (Flash Attention, mixed precision) significantly improve efficiency
- Understanding the theory helps debug and optimize in practice

This implementation serves as a template for various fine-tuning tasks and can be extended with additional features like multi-GPU training, advanced evaluation metrics, and production deployment strategies.
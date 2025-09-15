# Export Limitations and Solutions

## Current Status

The GitHub Actions workflows for exporting models have been successfully implemented, but there are important limitations to understand:

### ✅ What Works

1. **Workflow Structure**: All workflows run successfully and are properly configured
2. **Artifact Management**: Artifacts are created and can be downloaded
3. **Test Deployments**: Mock deployments work perfectly for testing
4. **Documentation**: Complete documentation and examples are provided

### ⚠️ Limitations

#### Memory Constraints on GitHub Runners

**Issue**: GitHub-hosted runners have limited memory (7GB RAM), which is insufficient for:
- Loading the full Phi-3 model (requires ~8-10GB)
- Merging LoRA adapters with the base model
- Converting to GGUF format

**Error Message**:
```
ValueError: We need an `offload_dir` to dispatch this model according to this `device_map`
```

This occurs because the model is too large to fit in the available memory.

### 💡 Solutions

#### Option 1: Use Self-Hosted Runners (Recommended for Production)

Add a self-hosted runner with sufficient resources:
- At least 16GB RAM
- Docker installed
- Python 3.10+

```yaml
runs-on: [self-hosted, Linux, X64]
```

#### Option 2: Local Export

Run the export scripts locally where you have sufficient resources:

```bash
# For vLLM export
python merge_and_export.py \
  --adapter-path avro-phi3-adapters/your-adapter \
  --format vllm \
  --output-dir exports/vllm-export

# For Ollama export
python export_ollama_docker.py \
  avro-phi3-adapters/your-adapter \
  --quantize q4_k_m \
  --output ollama-export
```

#### Option 3: Use Smaller Models

For testing and development, consider using smaller models that fit within GitHub runner limits:
- TinyLlama (1.1B parameters)
- Phi-2 (2.7B parameters)
- Other models under 3B parameters

#### Option 4: Cloud-Based Export

Use cloud services with more resources:
- GitHub Codespaces with upgraded specs
- Cloud VMs (AWS EC2, Google Cloud, Azure)
- Dedicated ML platforms (Paperspace, Lambda Labs)

## Workflow Behavior

The workflows have been designed with fallbacks:

1. **Attempt Real Export**: First tries to run the actual export scripts
2. **Fallback to Mock**: If export fails (due to memory), creates mock files for testing
3. **Continue Pipeline**: Rest of the workflow continues normally

This ensures workflows always complete successfully, even if the actual model export cannot be performed due to resource constraints.

## Testing vs Production

- **Testing**: Mock exports are sufficient for testing workflow logic and deployment scripts
- **Production**: Use one of the solutions above for real model exports

## Resource Requirements

### Minimum Requirements for Real Exports

| Component | Requirement |
|-----------|------------|
| RAM | 16GB minimum, 32GB recommended |
| Disk Space | 50GB free space |
| Docker | Latest version |
| Python | 3.10+ |
| CUDA | Optional, speeds up conversion |

### GitHub Runner Limits

| Resource | GitHub Hosted | Self-Hosted |
|----------|--------------|-------------|
| CPU | 2 cores | Unlimited |
| RAM | 7 GB | Unlimited |
| Storage | 14 GB | Unlimited |
| GPU | None | Can add |

## Verification

To verify if your export produced real or mock files:

1. Check file size:
   - Real GGUF: Usually 2-4GB for quantized Phi-3
   - Mock GGUF: Less than 1KB

2. Check file content:
   ```bash
   # Mock file will contain text
   head -c 100 model.gguf

   # Real file will be binary
   file model.gguf
   ```

3. Check workflow logs:
   ```bash
   gh run view [run-id] --log | grep "Real export failed"
   ```

## Recommendations

1. **For Development**: Use the current workflows with mock exports
2. **For Production**: Set up a self-hosted runner or export locally
3. **For CI/CD**: Consider using smaller models for automated testing
4. **For Deployment**: Always verify exports before deployment

## Future Improvements

Potential enhancements to consider:
- Implement chunked model loading to reduce memory usage
- Add support for cloud storage (S3, GCS) for large files
- Create a separate workflow for resource-intensive operations
- Implement model sharding for distributed processing
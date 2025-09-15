# Model Naming and Versioning Guide

This guide explains model naming conventions and version management for Hugging Face publishing, following ML community best practices.

## Overview

Based on 2024 research on Hugging Face naming conventions, **descriptive names are preferred over semantic versions** for ML models. The repository supports both approaches:

- **Descriptive naming** (recommended): `phi3mini4k-minimal-r32-a64-e20-20250914-132416-vllm-20250915-124051`
- **Version tags** (for reproducibility): Automated workflows triggered by git tags

## Supported Version Tag Formats

### 1. Semantic Versioning Tags (`v*.*.*`)
Standard semantic version format for model releases.

**Examples:**
- `v1.0.0` - Major model release
- `v1.1.0` - Minor improvements/features
- `v1.0.1` - Bug fixes/patches
- `v2.0.0` - Breaking changes/new architecture

### 2. Model-Specific Tags (`model-v*`)
Specific to model versions (distinct from code versions).

**Examples:**
- `model-v1.0.0` - First model version
- `model-v2.1.0` - Model iteration

### 3. Export-Specific Tags (`export-v*`)
For triggering exports with specific versions.

**Examples:**
- `export-v1.0.0` - Export-specific versioning

## Workflow Integration

### Automatic Triggers

Version tags automatically trigger these workflows:

1. **export-complete.yml** - Creates versioned export artifacts
2. **publish-huggingface.yml** - Publishes models to HF with version names

### Versioned Artifacts

When triggered by version tags, artifacts are named with versions:

**Without version tag:**
```
vllm-phi3mini4k-minimal-r32-a64-e20-20250915-132416.tar.gz
ollama-q4_k_m-phi3mini4k-minimal-r32-a64-e20-20250915-132416.tar.gz
```

**With version tag v2.0.0:**
```
vllm-phi3mini4k-minimal-r32-a64-e20-20250915-132416-v2.0.0.tar.gz
ollama-q4_k_m-phi3mini4k-minimal-r32-a64-e20-20250915-132416-v2.0.0.tar.gz
```

## Recommended Naming Conventions (2024 Best Practices)

### Descriptive Model Names (Preferred)

Following ML community research, descriptive names embed training metadata:

**Structure**: `<base>-<config>-<training-date>-<format>-<export-date>`

**Examples:**
- `phi3mini4k-minimal-r32-a64-e20-20250914-132416-vllm-20250915-124051`
- `phi3mini4k-minimal-r32-a64-e20-20250914-132416-q4_k_m-gguf-20250915-125110`
- `phi3mini4k-minimal-r32-a64-e20-20250914-132416-lora`

**Information Embedded:**
- `phi3mini4k`: Base model architecture
- `minimal`: Dataset/training approach
- `r32-a64-e20`: LoRA config (rank=32, alpha=64, epochs=20)
- `20250914-132416`: Training timestamp
- `vllm/q4_k_m/lora`: Export format
- `20250915-124051`: Export timestamp (when applicable)

### Why Descriptive Names Are Better

Recent 2024 research on Hugging Face found:

1. **More informative**: Embed training details directly in name
2. **Better discoverability**: Users can understand model without reading docs
3. **Community preference**: ML practitioners prefer metadata-rich names
4. **Avoid version confusion**: Semantic versions can be ambiguous for ML models

### Legacy Semantic Versioning (Still Supported)

For backward compatibility, semantic versions still work:

**With version tags:**
- `phi3mini4k-minimal-vllm-v2.0.1` (less descriptive)
- `phi3mini4k-minimal-q4_k_m-gguf-v2.0.1` (less descriptive)

## Usage Examples

### 1. Create and Push Version Tag

```bash
# Create a new semantic version tag
git tag v2.0.0 -m "Release v2.0.0: Improved AVRO schema generation"

# Push the tag to trigger workflows
git push origin v2.0.0
```

### 2. Model-Specific Versioning

```bash
# Tag for model-specific version
git tag model-v1.5.0 -m "Model v1.5.0: Enhanced training with new dataset"
git push origin model-v1.5.0
```

### 3. Export-Only Versioning

```bash
# Tag for export-specific version
git tag export-v1.0.0 -m "Export v1.0.0: Optimized quantization settings"
git push origin export-v1.0.0
```

### 4. Manual Publishing with Version Override

You can still trigger workflows manually and specify custom repo names:

```bash
# Manual trigger with custom name
gh workflow run publish-huggingface.yml \
  -f model_type=vllm \
  -f repo_name=phi3-avro-vllm-custom-v2.0.0
```

## Workflow Behavior

### Version Tag Triggered
When a version tag is pushed:

1. **export-complete.yml** automatically runs and:
   - Creates artifacts with version suffixes
   - Names export directories with versions
   - Generates versioned deployment packages

2. **publish-huggingface.yml** automatically runs and:
   - Publishes to HF with version-based repo names
   - Includes version information in model cards
   - Links artifacts to specific code versions

### Manual Dispatch
When manually triggered (no version tag):
- Uses timestamp-based naming (current behavior)
- Allows custom repo name specification
- Falls back to date-based suffixes

## Version Extraction Logic

The workflows extract version information using this priority:

1. **`v*.*.*`** → Extracts `2.0.0` from `v2.0.0`
2. **`model-v*`** → Extracts `1.5.0` from `model-v1.5.0`
3. **`export-v*`** → Extracts `1.0.0` from `export-v1.0.0`
4. **Other tags** → Uses full tag name as version

## Best Practices

### Version Naming Convention

```bash
# Semantic versioning for model releases
v1.0.0    # First stable model
v1.1.0    # Improved performance/features
v1.0.1    # Bug fixes/small improvements
v2.0.0    # Major changes (new architecture, breaking changes)

# Model-specific for iterations
model-v1.0.0   # First model iteration
model-v1.1.0   # Improved model with same architecture
model-v2.0.0   # New model architecture

# Export-specific for deployment versions
export-v1.0.0  # First export configuration
export-v1.1.0  # Optimized export settings
```

### Tagging Workflow

1. **Test first** - Always test workflows manually before tagging
2. **Update CHANGELOG** - Document changes before tagging
3. **Clean working directory** - Ensure no uncommitted changes
4. **Descriptive messages** - Use meaningful tag messages

```bash
# Good tagging workflow
git status                    # Check clean working directory
git log --oneline -5          # Review recent changes
git tag v2.0.0 -m "Release v2.0.0:
- Improved AVRO schema accuracy by 15%
- Added support for nested schemas
- Fixed quantization issues in GGUF export
- Updated model cards with performance metrics"
git push origin v2.0.0
```

### Release Management

1. **Pre-release testing** - Use manual triggers to test
2. **Version planning** - Plan version increments
3. **Documentation** - Update docs with new versions
4. **HF repository management** - Monitor published models

## Monitoring Versioned Releases

### Check Tag-Triggered Workflows

```bash
# List recent workflow runs triggered by tags
gh run list --event=push --limit=10

# View specific tag-triggered run
gh run view <run-id> --log
```

### Verify Published Models

```bash
# Check your Hugging Face models
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/models?author=yourusername
```

### List Version Tags

```bash
# Show all version tags
git tag -l "v*" --sort=-version:refname

# Show model-specific tags
git tag -l "model-v*" --sort=-version:refname

# Show recent tags with dates
git for-each-ref --format="%(refname:short) %(creatordate)" refs/tags | sort -k2
```

## Troubleshooting

### Common Issues

1. **Tag not triggering workflow**
   ```bash
   # Ensure tag matches pattern exactly
   git tag -l | grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$"

   # Check if tag was pushed
   git ls-remote --tags origin
   ```

2. **Version not extracted correctly**
   - Check workflow logs for "Tagged release detected" message
   - Ensure tag follows supported formats
   - Verify no typos in tag names

3. **Artifacts not versioned**
   - Check that version extraction step succeeded
   - Verify job dependencies in workflow
   - Look for version outputs in logs

### Debug Version Extraction

Add debug logging to see version extraction:

```bash
# Check what version would be extracted
TAG_NAME="v2.0.0"
if [[ $TAG_NAME == v*.*.* ]]; then
  VERSION=${TAG_NAME#v}
  echo "Extracted version: $VERSION"
fi
```

## Migration from Date-Based Naming

### Current State (Date-Based)
- Artifacts: `vllm-adapter-20250915`
- HF repos: `phi3-avro-vllm-20250915`

### After Version Tags
- Artifacts: `vllm-adapter-v2.0.0`
- HF repos: `phi3-avro-vllm-v2.0.0`

### Gradual Migration
1. **Start with exports** - Tag exports first to test
2. **Publish versioned models** - Use version tags for HF publishing
3. **Maintain compatibility** - Manual triggers still work with timestamps
4. **Update documentation** - Reference versioned models in docs

## Future Enhancements

Potential improvements to the versioning system:

1. **Auto-increment versioning** - Automatically bump versions
2. **Release notes generation** - Auto-generate from commits
3. **Version compatibility matrix** - Track model compatibility
4. **Rollback support** - Easy reversion to previous versions
5. **Multi-format versioning** - Different versions per format (vLLM vs Ollama)

## Example: Complete Versioning Workflow

```bash
# 1. Complete development work
git add .
git commit -m "Improve AVRO schema generation accuracy"

# 2. Test manually first
gh workflow run export-complete.yml -f export_formats=vllm

# 3. Create and push version tag
git tag v2.1.0 -m "v2.1.0: Improved accuracy and performance"
git push origin v2.1.0

# 4. Monitor automatic workflows
gh run list --event=push --limit=5

# 5. Verify published models
# Check https://huggingface.co/yourusername/phi3-avro-vllm-v2.1.0
```

This versioning system ensures that every model release is:
- **Traceable** to specific code versions
- **Reproducible** with exact same configurations
- **Organized** with clear version naming
- **Automated** through git tag triggers
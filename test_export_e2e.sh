#!/bin/bash

echo "=========================================="
echo "End-to-End Export Test"
echo "=========================================="

ADAPTER_PATH="avro-phi3-adapters/phi3mini4k-minimal-r32-a64-e20-20250914-132416"

# Test Ollama export (faster than vLLM)
echo "Running Ollama export with q4_k_m quantization..."
uv run python export_ollama_docker.py "$ADAPTER_PATH" --quantize q4_k_m

# Check if export was successful
EXPORT_DIR=$(ls -td exports/*ollama-docker* 2>/dev/null | head -1)
if [ -n "$EXPORT_DIR" ] && [ -d "$EXPORT_DIR" ]; then
    echo "✅ Export successful: $EXPORT_DIR"
    echo "Contents:"
    ls -lh "$EXPORT_DIR"

    # Test artifact creation
    echo ""
    echo "Creating test artifact..."
    cd "$EXPORT_DIR"
    tar -czf ../test-ollama-artifact.tar.gz . 2>/dev/null && {
        echo "✅ Artifact created successfully"
        ls -lh ../test-ollama-artifact.tar.gz
    } || echo "❌ Failed to create artifact"
else
    echo "❌ Export failed or directory not found"
    exit 1
fi
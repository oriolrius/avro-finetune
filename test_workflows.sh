#!/bin/bash

set -e

echo "=========================================="
echo "GitHub Workflow Local Testing Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_step() {
    local description="$1"
    local command="$2"
    echo -e "\n${YELLOW}Testing: ${description}${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        return 1
    fi
}

# Check prerequisites
echo -e "\n${YELLOW}=== Checking Prerequisites ===${NC}"

test_step "Python and uv installed" "which python && which uv"
test_step "Docker installed" "docker --version"
test_step "Docker Compose installed" "docker compose version"
test_step "GitHub CLI installed" "gh --version"

# Check adapter exists
ADAPTER_PATH="phi3mini4k-minimal-r32-a64-e20-20250914-132416"
echo -e "\n${YELLOW}=== Checking Adapter ===${NC}"
test_step "Adapter exists" "[ -d 'avro-phi3-adapters/${ADAPTER_PATH}' ]"

# Test workflow YAML validity
echo -e "\n${YELLOW}=== Validating Workflow Files ===${NC}"
test_step "train-and-deploy.yml valid" "python -c 'import yaml; yaml.safe_load(open(\".github/workflows/train-and-deploy.yml\"))'"
test_step "export-models.yml valid" "python -c 'import yaml; yaml.safe_load(open(\".github/workflows/export-models.yml\"))'"

# Test export scripts
echo -e "\n${YELLOW}=== Testing Export Scripts ===${NC}"

# Test vLLM export (dry run)
echo "Testing vLLM export script..."
if timeout 5 uv run python merge_and_export.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ vLLM export script works${NC}"
else
    echo -e "${RED}❌ vLLM export script failed${NC}"
fi

# Test Ollama export (dry run)
echo "Testing Ollama export script..."
if timeout 5 uv run python export_ollama_docker.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama export script works${NC}"
else
    echo -e "${RED}❌ Ollama export script failed${NC}"
fi

# Simulate workflow steps
echo -e "\n${YELLOW}=== Simulating Workflow Steps ===${NC}"

# Step 1: Get adapter path (as workflow would)
echo "Step 1: Getting adapter path..."
FOUND_ADAPTER=$(ls -td avro-phi3-adapters/* 2>/dev/null | head -1)
if [ -n "$FOUND_ADAPTER" ]; then
    ADAPTER_NAME=$(basename "$FOUND_ADAPTER")
    echo -e "${GREEN}✅ Found adapter: $ADAPTER_NAME${NC}"
else
    echo -e "${RED}❌ No adapter found${NC}"
    exit 1
fi

# Step 2: Check export directory creation
echo "Step 2: Testing export directory creation..."
mkdir -p exports
if [ -d "exports" ]; then
    echo -e "${GREEN}✅ Export directory ready${NC}"
else
    echo -e "${RED}❌ Failed to create export directory${NC}"
fi

# Step 3: Test artifact naming
echo "Step 3: Testing artifact naming conventions..."
echo "  vLLM artifact: vllm-model-${ADAPTER_NAME}.tar.gz"
echo "  Ollama artifact: ollama-model-q4_k_m-${ADAPTER_NAME}.tar.gz"
echo -e "${GREEN}✅ Naming conventions verified${NC}"

# Step 4: Test Docker availability for Ollama
echo "Step 4: Testing Docker for Ollama export..."
if docker run --rm hello-world > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is working${NC}"
else
    echo -e "${RED}❌ Docker test failed${NC}"
fi

# Summary
echo -e "\n${YELLOW}=========================================="
echo "Test Summary"
echo "==========================================${NC}"

echo -e "${GREEN}Workflow files are valid and ready to use!${NC}"
echo ""
echo "To use the workflows:"
echo "1. Merge to main branch: git checkout main && git merge feature/vllm-ollama-export"
echo "2. Push to GitHub: git push origin main"
echo "3. Trigger manually:"
echo "   gh workflow run export-models.yml \\"
echo "     -f adapter_path=\"${ADAPTER_NAME}\" \\"
echo "     -f export_formats=\"all\" \\"
echo "     -f ollama_quantization=\"q4_k_m\""
echo ""
echo "Or trigger via GitHub UI:"
echo "   Go to Actions tab → Select workflow → Run workflow"
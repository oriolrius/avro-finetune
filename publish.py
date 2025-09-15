#!/usr/bin/env python
"""
Unified publishing script for Hugging Face Hub.
Handles both LoRA adapters and full models (vLLM/Ollama exports).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required packages not installed")
    print("Run: uv pip install huggingface-hub python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


def publish_to_hub(
    path: str,
    repo_name: Optional[str] = None,
    organization: Optional[str] = None,
    private: bool = False,
    repo_type: str = "model"
) -> str:
    """
    Publish model or dataset to Hugging Face Hub.

    Args:
        path: Path to directory to upload
        repo_name: Repository name (auto-generated if not provided)
        organization: HF organization (uses personal account if not provided)
        private: Whether to create private repository
        repo_type: Type of repository ("model" or "dataset")

    Returns:
        URL of the published repository
    """

    # Validate token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found in .env file")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Validate path
    path = Path(path)
    if not path.exists():
        print(f"❌ Error: Path not found: {path}")
        sys.exit(1)

    # Auto-generate repo name if not provided
    if not repo_name:
        if "adapter" in path.name or "lora" in path.name.lower():
            repo_name = f"phi3-lora-{path.name}"
        elif "vllm" in path.name:
            repo_name = f"phi3-vllm-{datetime.now().strftime('%Y%m%d')}"
        elif "ollama" in path.name or "gguf" in str(path):
            repo_name = f"phi3-gguf-{datetime.now().strftime('%Y%m%d')}"
        else:
            repo_name = f"phi3-model-{datetime.now().strftime('%Y%m%d')}"

    # Initialize API
    api = HfApi(token=token)

    # Determine full repo ID
    repo_id = f"{organization}/{repo_name}" if organization else repo_name

    print(f"📤 Publishing to Hugging Face Hub")
    print(f"   Repository: {repo_id}")
    print(f"   Type: {repo_type}")
    print(f"   Private: {private}")
    print(f"   Path: {path}")

    try:
        # Create repository
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type=repo_type,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_url}")

        # Create README if it doesn't exist
        readme_path = path / "README.md"
        if not readme_path.exists():
            print("📝 Creating README.md")
            readme_content = f"""# {repo_name}

## Model Information
- **Base Model**: microsoft/Phi-3-mini-4k-instruct
- **Type**: {"LoRA Adapter" if "adapter" in path.name else "Fine-tuned Model"}
- **Created**: {datetime.now().strftime('%Y-%m-%d')}

## Usage

### For LoRA Adapters
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "{repo_id}")
```

### For GGUF Models (Ollama)
```bash
ollama run {repo_id}
```

### For vLLM
```bash
docker run --gpus all -p 8000:8000 \\
    vllm/vllm-openai:latest \\
    --model {repo_id}
```

## Training Details
Generated from fine-tuning project: https://github.com/oriolrius/avro-finetune
"""
            readme_path.write_text(readme_content)

        # Upload files
        print("📦 Uploading files...")

        # Check for large files
        large_files = []
        for file in path.rglob("*"):
            if file.is_file() and file.stat().st_size > 50 * 1024 * 1024:  # 50MB
                large_files.append(file.name)

        if large_files:
            print(f"   Large files detected: {', '.join(large_files)}")
            print("   Using chunked upload for better reliability...")

        # Upload folder
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Upload {path.name}"
        )

        print(f"✅ Successfully published to: {repo_url}")
        return repo_url

    except Exception as e:
        print(f"❌ Error publishing to Hugging Face: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Publish models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish latest adapter
  uv run python publish.py --latest --type adapter

  # Publish specific export
  uv run python publish.py exports/my-model-vllm

  # Publish to organization
  uv run python publish.py exports/my-model --org myorg --private
        """
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to model/adapter directory"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest export/adapter"
    )
    parser.add_argument(
        "--type",
        choices=["adapter", "export", "auto"],
        default="auto",
        help="Type of model to publish"
    )
    parser.add_argument(
        "--name",
        help="Repository name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--org",
        help="Hugging Face organization"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Publish as dataset instead of model"
    )

    args = parser.parse_args()

    # Determine path
    if args.latest:
        if args.type == "adapter" or (args.type == "auto" and not args.path):
            # Find latest adapter
            adapter_dir = Path("avro-phi3-adapters")
            if adapter_dir.exists():
                adapters = sorted([d for d in adapter_dir.iterdir() if d.is_dir()])
                if adapters:
                    path = adapters[-1]
                    print(f"Using latest adapter: {path.name}")
                else:
                    print("❌ No adapters found")
                    sys.exit(1)
            else:
                print("❌ Adapter directory not found")
                sys.exit(1)
        else:
            # Find latest export
            export_dir = Path("exports")
            if export_dir.exists():
                exports = sorted([d for d in export_dir.iterdir() if d.is_dir()])
                if exports:
                    path = exports[-1]
                    print(f"Using latest export: {path.name}")
                else:
                    print("❌ No exports found")
                    sys.exit(1)
            else:
                print("❌ Export directory not found")
                sys.exit(1)
    elif args.path:
        path = args.path
    else:
        parser.print_help()
        sys.exit(1)

    # Get organization from env if not provided
    if not args.org:
        args.org = os.getenv("HF_ORGANIZATION")

    # Publish
    url = publish_to_hub(
        path=path,
        repo_name=args.name,
        organization=args.org,
        private=args.private,
        repo_type="dataset" if args.dataset else "model"
    )

    print(f"\n🎉 Published successfully!")
    print(f"   View at: {url}")


if __name__ == "__main__":
    main()
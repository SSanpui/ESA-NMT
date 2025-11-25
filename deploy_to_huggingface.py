#!/usr/bin/env python3
"""
Deployment script for uploading models to Hugging Face Hub

Usage:
    python deploy_to_huggingface.py --model_type nllb --translation_pair bn-hi --hf_username your_username
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo
import torch

def deploy_model(model_dir: str, hf_username: str, model_type: str, translation_pair: str):
    """Deploy model to Hugging Face Hub"""

    repo_name = f"emotion-semantic-nmt-{model_type}-{translation_pair}"
    repo_id = f"{hf_username}/{repo_name}"

    print(f"🚀 Deploying model to Hugging Face Hub...")
    print(f"   Repository: {repo_id}")

    # Initialize API
    api = HfApi()

    try:
        # Create repository
        print(f"📦 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )

        # Upload files
        print(f"📤 Uploading files from {model_dir}...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model"
        )

        print(f"✅ Model deployed successfully!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"❌ Error deploying model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy ESA-NMT models to Hugging Face Hub")
    parser.add_argument("--model_type", type=str, required=True, choices=['nllb', 'indictrans2'],
                       help="Model type")
    parser.add_argument("--translation_pair", type=str, required=True, choices=['bn-hi', 'bn-te'],
                       help="Translation pair")
    parser.add_argument("--hf_username", type=str, required=True,
                       help="Hugging Face username")
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Model directory (auto-detected if not provided)")

    args = parser.parse_args()

    # Auto-detect model directory if not provided
    if args.model_dir is None:
        args.model_dir = f"./models/emotion-semantic-nmt-{args.model_type}-{args.translation_pair}"

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        print("   Please run model training and preparation first.")
        return

    # Deploy
    deploy_model(args.model_dir, args.hf_username, args.model_type, args.translation_pair)

if __name__ == "__main__":
    main()

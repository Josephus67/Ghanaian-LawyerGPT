#!/usr/bin/env python3
"""
Hugging Face Dataset Uploader for Ghanaian LawyerGPT

This script prepares and uploads the Ghanaian law dataset to Hugging Face Hub.

Prerequisites:
    pip install datasets huggingface-hub

Usage:
    1. Login to Hugging Face: huggingface-cli login
    2. Run: python upload_to_huggingface.py

Configuration:
    - Change DATASET_NAME to your desired repository name
    - Change DATASET_FILES to include your dataset files
"""

import os
import json
from typing import List, Dict

try:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Required libraries not installed.")
    print("Install with: pip install datasets huggingface-hub")


# Configuration - Change these values
DATASET_NAME = "Ghanaian_Law_QA"  # Change to your desired dataset name
DATASET_FILES = [
    "ghanaian_law_comprehensive.jsonl",
    "ghanaian_law_dataset_sample.jsonl",
]
DATASET_DESCRIPTION = """
# Ghanaian Law Question-Answer Dataset

A comprehensive dataset of question-answer pairs about Ghanaian law for fine-tuning language models.

## Dataset Description

This dataset contains Q&A pairs covering:

- **Constitutional Law**: The 1992 Constitution of Ghana
- **Criminal Law**: Criminal Offences Act, 1960 (Act 29)
- **Labour Law**: Labour Act, 2003 (Act 651)
- **Land Law**: Land Act, 2020 (Act 1036)
- **Company Law**: Companies Act, 2019 (Act 992)
- **Family Law**: Marriage laws, Intestate Succession Law
- **Data Protection**: Data Protection Act, 2012 (Act 843)
- **Environmental Law**: Environmental Protection Agency Act, 1994
- **Right to Information**: Right to Information Act, 2019 (Act 989)
- **Court System**: Structure and jurisdiction of Ghanaian courts

## Format

Each entry contains:
- `question`: A legal question about Ghanaian law
- `answer`: A detailed answer explaining the relevant law

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("your-username/Ghanaian_Law_QA")
```

## Citation

If you use this dataset, please cite:

```
@dataset{ghanaian_law_qa,
  title={Ghanaian Law Question-Answer Dataset},
  year={2024},
  publisher={Hugging Face}
}
```

## License

This dataset is provided for educational and research purposes.
"""


def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_dataset(data_dir: str) -> Dataset:
    """Prepare the dataset from JSONL files."""
    all_data = []
    
    for filename in DATASET_FILES:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            data = load_jsonl(filepath)
            all_data.extend(data)
            print(f"  Loaded {len(data)} entries")
        else:
            print(f"Warning: {filepath} not found")
    
    print(f"\nTotal entries: {len(all_data)}")
    
    # Deduplicate based on question
    seen_questions = set()
    unique_data = []
    for entry in all_data:
        q = entry.get("question", "")
        if q not in seen_questions:
            seen_questions.add(q)
            unique_data.append(entry)
    
    print(f"Unique entries after deduplication: {len(unique_data)}")
    
    # Create dataset
    dataset = Dataset.from_dict({
        "question": [d["question"] for d in unique_data],
        "answer": [d["answer"] for d in unique_data]
    })
    
    return dataset


def upload_dataset(dataset: Dataset, repo_name: str, username: str = None):
    """Upload the dataset to Hugging Face Hub."""
    if username:
        repo_id = f"{username}/{repo_name}"
    else:
        repo_id = repo_name
    
    print(f"\nUploading to: {repo_id}")
    
    # Create dataset dict with train split
    dataset_dict = DatasetDict({"train": dataset})
    
    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        private=False,
        commit_message="Upload Ghanaian Law Q&A dataset"
    )
    
    print(f"\nDataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def create_dataset_card(data_dir: str):
    """Create a README.md dataset card."""
    card_path = os.path.join(data_dir, "README_DATASET.md")
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(DATASET_DESCRIPTION)
    print(f"Dataset card created: {card_path}")


def main():
    """Main entry point."""
    if not HF_AVAILABLE:
        print("Please install required libraries first:")
        print("  pip install datasets huggingface-hub")
        return
    
    # Get script directory and data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "dataset")
    
    print("=" * 60)
    print("Ghanaian Law Dataset - Hugging Face Uploader")
    print("=" * 60)
    
    # Check if logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info.get("name", None)
        print(f"\nLogged in as: {username}")
    except Exception:
        print("\nNot logged in to Hugging Face.")
        print("Please run: huggingface-cli login")
        print("\nContinuing with dataset preparation only...\n")
        username = None
    
    # Prepare dataset
    print("\n1. Preparing dataset...")
    dataset = prepare_dataset(data_dir)
    
    # Show sample
    print("\n2. Sample entries:")
    print("-" * 40)
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        print(f"\nQ: {example['question'][:100]}...")
        print(f"A: {example['answer'][:100]}...")
    
    # Create dataset card
    print("\n3. Creating dataset card...")
    create_dataset_card(data_dir)
    
    # Upload if logged in
    if username:
        print("\n4. Uploading to Hugging Face Hub...")
        try:
            upload_dataset(dataset, DATASET_NAME, username)
        except Exception as e:
            print(f"Upload failed: {e}")
            print("You can upload manually using the Hugging Face web interface.")
    else:
        print("\n4. Skipping upload (not logged in)")
        print("   To upload later, run: huggingface-cli login")
        print("   Then run this script again.")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

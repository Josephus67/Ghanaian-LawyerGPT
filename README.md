# Ghanaian Law AI: Fine-Tuning Falcon-7B & LLAMA 2 Language Models

Welcome to this project where we are adapting two cutting-edge language models, Falcon-7B & LLAMA 2, to become proficient in Ghanaian law.

## Overview

This AI legal project aims to create a language model fine-tuned on Ghanaian law, including the Constitution of Ghana (1992), statutory laws, case law, and legal principles specific to Ghana's legal system. The project combines:

- **Falcon-7B & LLAMA 2**: State-of-the-art language models, prepped and ready for legal training.
- **PEFT & QLoRA**: The dream duo for memory-efficient and high-performance model fine-tuning.
- **Comprehensive Dataset**: 100+ Q&A pairs covering Ghanaian constitutional law, criminal law, land law, family law, and more!

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: at least 16GB VRAM)
- Hugging Face account (for model access)

### Installation

```bash
pip install -r requirements.txt
```

### Generate More Data

Run the scraping script to generate the comprehensive dataset:

```bash
cd scripts
python scrape_ghanaian_law.py
```

This will generate `dataset/ghanaian_law_comprehensive.jsonl` with 100+ Q&A pairs.

### Upload to Hugging Face

To upload the dataset to Hugging Face Hub:

```bash
# Login to Hugging Face
huggingface-cli login

# Run the upload script
cd scripts
python upload_to_huggingface.py
```

## Dataset

### Dataset Files

| File | Description | Q&A Pairs |
|------|-------------|-----------|
| `ghanaian_law_comprehensive.jsonl` | Main dataset with constitutional and statutory law | 107 |
| `ghanaian_law_dataset_sample.jsonl` | Sample dataset for quick testing | 20 |

### Dataset Format

The dataset uses a question-answer format in JSONL:

```json
{"question": "What does Article 1 of the 1992 Constitution of Ghana say about supremacy of the constitution?", "answer": "Article 1 of the 1992 Constitution of Ghana provides that..."}
```

### Legal Topics Covered

- **Constitutional Law** (1992 Constitution of Ghana)
  - Supremacy and enforcement of the Constitution
  - Fundamental human rights and freedoms (Articles 12-33)
  - The Executive (President, Vice-President)
  - The Legislature (Parliament)
  - The Judiciary (Supreme Court, Court of Appeal, High Court)
  - Chieftaincy institution
  - Commission on Human Rights and Administrative Justice (CHRAJ)

- **Criminal Law** (Criminal Offences Act, 1960 - Act 29)
  - Murder, manslaughter, and assault
  - Theft, robbery, and property crimes
  - Sexual offences
  - Penalties and sentencing

- **Labour Law** (Labour Act, 2003 - Act 651)
  - Employment contracts and termination
  - Working hours and leave entitlements
  - Collective bargaining and trade unions
  - Unfair dismissal

- **Land Law** (Land Act, 2020 - Act 1036)
  - Types of land interests (allodial, usufructuary, leasehold)
  - Customary land management
  - Land registration
  - Role of Lands Commission

- **Company Law** (Companies Act, 2019 - Act 992)
  - Company incorporation
  - Types of companies
  - Directors' duties
  - Winding up

- **Family Law**
  - Types of marriages (customary, ordinance, Islamic)
  - Intestate Succession Law, 1985
  - Matrimonial Causes Act, 1971

- **Other Laws**
  - Data Protection Act, 2012 (Act 843)
  - Environmental Protection Agency Act, 1994 (Act 490)
  - Right to Information Act, 2019 (Act 989)

## Training

Use the notebooks in `Training_Code/` to fine-tune the models:

1. `Training_LawyerGPT_Finetune_falcon7b_Ghanaian_Law_Data.ipynb` - For Falcon-7B fine-tuning
2. `Nisaar__Llama_2_Fine_Tuning_Using_QLora.ipynb` - For LLAMA 2 fine-tuning

## Inference

Use the scripts in `Inference_Code/` to run inference with the fine-tuned models.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/scrape_ghanaian_law.py` | Generates the comprehensive Ghanaian law dataset |
| `scripts/upload_to_huggingface.py` | Uploads the dataset to Hugging Face Hub |
| `scripts/datasetformatter.py` | Formats datasets for training |

## Fine Tuning Process

The fine-tuning process uses PEFT (Parameter-Efficient Fine-Tuning) with QLoRA (Quantized Low-Rank Adaptation) to efficiently train the model on Ghanaian legal data while minimizing GPU memory requirements.

## Track the Progress

Get a front-row seat to the training progress with TensorBoard. Kickstart it, navigate to the provided localhost link, and witness the models learn.

## Credits

This project is adapted from the [Indian-LawyerGPT](https://github.com/NisaarAgharia/Indian-LawyerGPT) project by Nisaar Agharia. Special thanks for providing the foundational architecture and methodology.

# Ghanaian Law AI: Fine-Tuning Falcon-7B & LLAMA 2 Language Models

Welcome to this project where we are adapting two cutting-edge language models, Falcon-7B & LLAMA 2, to become proficient in Ghanaian law.

## Overview

This AI legal project aims to create a language model fine-tuned on Ghanaian law, including the Constitution of Ghana (1992), statutory laws, case law, and legal principles specific to Ghana's legal system. The project combines:

- **Falcon-7B & LLAMA 2**: State-of-the-art language models, prepped and ready for legal training.
- **PEFT & QLoRA**: The dream duo for memory-efficient and high-performance model fine-tuning.
- **Custom Dataset**: Ghanaian law knowledge, spanning constitutional law, criminal law, land law, family law, and more!

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: at least 16GB VRAM)
- Hugging Face account (for model access)

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Format

The dataset uses a question-answer format in JSONL. See `dataset/ghanaian_law_dataset_sample.jsonl` for examples:

```json
{"question": "Your legal question about Ghanaian law", "answer": "The detailed answer"}
```

## Training

Use the notebooks in `Training_Code/` to fine-tune the models:

1. `Training_LawyerGPT_Finetune_falcon7b_Ghanaian_Law_Data.ipynb` - For Falcon-7B fine-tuning
2. `Nisaar__Llama_2_Fine_Tuning_Using_QLora.ipynb` - For LLAMA 2 fine-tuning

## Inference

Use the scripts in `Inference_Code/` to run inference with the fine-tuned models.

## Dataset Structure

Our dataset is designed with two key features: `question` and `answer`. The dataset covers:

- **Constitutional Law**: The 1992 Constitution of Ghana, fundamental rights, and government structure
- **Criminal Law**: Criminal Offences Act, 1960 (Act 29), criminal procedures
- **Land Law**: Land Act, 2020 (Act 1036), customary land rights
- **Family Law**: Marriage laws, inheritance, and succession
- **Commercial Law**: Companies Act, 2019 (Act 992), contract law
- **Labor Law**: Labour Act, 2003 (Act 651)

## Fine Tuning Process

The fine-tuning process uses PEFT (Parameter-Efficient Fine-Tuning) with QLoRA (Quantized Low-Rank Adaptation) to efficiently train the model on Ghanaian legal data while minimizing GPU memory requirements.

## Track the Progress

Get a front-row seat to the training progress with TensorBoard. Kickstart it, navigate to the provided localhost link, and witness the models learn:

## Credits

This project is adapted from the [Indian-LawyerGPT](https://github.com/NisaarAgharia/Indian-LawyerGPT) project by Nisaar Agharia. Special thanks for providing the foundational architecture and methodology.

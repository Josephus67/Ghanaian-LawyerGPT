# -*- coding: utf-8 -*-
"""Inference_LawyerGPT_Finetune_falcon7b_Ghanaian_Law_Data.py

Inference script for Ghanaian Law GPT model fine-tuned on Ghanaian legal data.

This script demonstrates how to run inference with a fine-tuned Falcon-7B model
for answering questions about Ghanaian law.

### Install requirements

First, run the cells below to install the requirements:
"""

!nvidia-smi

!pip install -Uqqq pip --progress-bar off
!pip install -qqq bitsandbytes==0.39.0
!pip install -qqq torch==2.0.1 --progress-bar off
!pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc --progress-bar off
!pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f --progress-bar off
!pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71 --progress-bar off
!pip install -qqq datasets==2.12.0 --progress-bar off
!pip install -qqq loralib==0.1.1 --progress-bar off
!pip install einops

import os
# from pprint import pprint
# import json

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

notebook_login()
#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

"""### Load dataset"""

from datasets import load_dataset

# TODO: Replace with your Ghanaian law dataset on Hugging Face
# dataset_name = "your-username/Ghanaian_Law_Dataset"
# For now, using local dataset file: dataset/ghanaian_law_dataset_sample.jsonl
dataset_name = "nisaar/Lawyer_GPT_India"  # Replace with Ghanaian law dataset
#dataset_name = "patrick11434/TEST_LLM_DATASET"
dataset = load_dataset(dataset_name, split="train")

"""## Load adapters from the Hub

You can also directly load adapters from the Hub using the commands below:
"""

from peft import *

#change peft_model_id
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# TODO: Replace with your fine-tuned Ghanaian law model on Hugging Face
# peft_model_id = "your-username/falcon7b-Ghanaian_Law"
peft_model_id = "nisaar/falcon7b-Indian_Law_150Prompts"  # Replace with Ghanaian law model
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token


model = PeftModel.from_pretrained(model, peft_model_id)

"""## Inference

You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference as you would do it usually in `transformers`.
"""

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config_temperature = 1
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config_eod_token_id = tokenizer.eos_token_id

DEVICE = "cuda:0"

# Commented out IPython magic to ensure Python compatibility.
# %%time
# prompt = f"""
# <human>: Who appoints the Chief Justice of India?
# <assistant>:
# """.strip()
# 
# encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
# with torch.inference_mode():
#   outputs = model.generate(
#       input_ids=encoding.attention_mask,
#       generation_config=generation_config,
#   )
# print(tokenizer.decode(outputs[0],skip_special_tokens=True))

def generate_response(question: str) -> str:
    prompt = f"""
    <human>: {question}
    <assistant>:
    """.strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0],skip_special_tokens=True)

    assistant_start = '<assistant>:'
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start):].strip()

prompt = "Explain the structure of Ghana's Parliament under the 1992 Constitution"
print(generate_response(prompt))

prompt = "What are the duties of the President of Ghana as per the Constitution?"
print(generate_response(prompt))

prompt = "Write a legal memo on the protection of fundamental human rights under Chapter 5 of the 1992 Constitution of Ghana."
print(generate_response(prompt))

prompt = "Explain the concept of 'Separation of Powers' in the 1992 Constitution of Ghana"
print(generate_response(prompt))

prompt = "Can you explain the steps for registration of a trademark in Ghana?"
print(generate_response(prompt))

prompt = "What are the potential implications of the Data Protection Act, 2012 (Act 843) on tech companies in Ghana?"
print(generate_response(prompt))

prompt = "Can you draft a non-disclosure agreement (NDA) under Ghanaian law?"
print(generate_response(prompt))

prompt = "Can you summarize the main points of Article 13 (Right to Life) of the 1992 Constitution of Ghana?"
print(generate_response(prompt))

prompt = "Can you summarize the main arguments in a landmark Supreme Court of Ghana judgment?"
print(generate_response(prompt))

prompt = "What is the role of the Commission on Human Rights and Administrative Justice (CHRAJ) in Ghana?"
print(generate_response(prompt))

prompt = "What is the role of CHRAJ in investigating corruption cases in Ghana?"
print(generate_response(prompt))

prompt = "Can you draft a confidentiality clause for a contract under Ghanaian law?"
print(generate_response(prompt))

prompt = "How are Directive Principles of State Policy enshrined in the 1992 Constitution of Ghana?"
print(generate_response(prompt))

prompt = "What is the role of the Supreme Court of Ghana in preserving the fundamental rights of citizens?"
print(generate_response(prompt))

prompt = "Analyze the potential impact of the Right to Information Act, 2019 (Act 989) for citizens in Ghana"
print(generate_response(prompt))

prompt = "Analyze the potential impact of digital rights for citizens in Ghana"
print(generate_response(prompt))

prompt = "Discuss the potential effects of a Universal Basic Income policy in Ghana"
print(generate_response(prompt))

prompt = "Analyze the potential impact of the Free Senior High School policy in Ghana"
print(generate_response(prompt))
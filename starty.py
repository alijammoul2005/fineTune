from huggingface_hub import login
login("")

#=========================================================================================================

#load the model
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

model_config = BitsAndBytesConfig( load_in_8bit = True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config = model_config,
    device_map = 'auto'
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

#=========================================================================================================

#prepare model for training + LORA
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model.gradient_checkpoint_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r             = 16,
    lora_alpha    = 32,
    target_module = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    lora_dropout  = 0.05,
    bias          = "none",
    task_type     = "CAUSAL_LM"
)
lora_model = get_peft_model(model, lora_config)

#===========================================================================================================

#load and preprocess the data
import transformers
import pandas as pd
from datasets import Dataset

fake_df = pd.read_csv("Path")
true_df = pd.read_csv("path")

fake_df['label'] = 0
true_df['label'] = 1

df      = pd.concat([fake_df,true_df]).sample(frac=1).reset_index(drop=True)
dataset = Dataset.from_pandas(df)
data    = dataset.train_test_split(test_size=0.3,seed=200)

def formate(info):
    ans   = 'True' if info['label'] == 1 else 'False'
    promt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a news validator. Classify the input as REAL or FAKE.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Article: {info['text']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{ans}<|eot_id|>"
    )
#   promt = [
#         {"role": "system", "content": "You are a news validator. Classify as REAL or FAKE."},
#         {"role": "user", "content": f"Article: {example['text']}"},
#         {"role": "assistant", "content": ans}
#     ]
#     formatted_prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=False
#     )
    return {"promt": promt}
data['train'] = data['train'].map(formate)
data['test'] = data['test'].map(formate)

#===========================================================================================================

#train the model
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    dataset_text_field          = "promt",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    warmup_steps                = 50,
    learning_rate               = 3e-4,
    bf16                        = True,
    logging_steps               = 1,
    output_dir                  = 'outputs',
    remove_unused_columns       = True,
    report_to                   = "none"
)

coach = SFTTrainer(
    model         = lora_model,
    train_dataset = data['train'],
    eval_dataset  = data['test'],
    args          = sft_config,
)

coach.train()

#===========================================================================================================
#export the model

from peft import PeftModel

name = "meta-llama/Llama-3.2-3B"
checkpoint_path = "/kaggle/working/outputs/checkpoint-264"
hf_repo_id = "Ali-jammoul/fake-news-detector-3b"

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,  # Full precision
    device_map="cpu",
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
merged_model = lora_model.merge_and_unload()

merged_model.save_pretrained("./final_merged_model")
# lora_model.save_pretrained("./final_lora_model")
tokenizer.save_pretrained("./final_merged_model")

merged_model.push_to_hub(hf_repo_id, use_auth_token=True)
tokenizer.push_to_hub(hf_repo_id, use_auth_token=True)




from dataclasses import dataclass

import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers.models.clap.convert_clap_original_pytorch_to_hf import processor
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


class DataEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_and_split(self, fake_path, true_path, test_size=0.3):
        # fake_df = pd.read_csv(fake_path).assign(label=0)
        # true_df = pd.read_csv(true_path).assign(label=1)
        # df = pd.concat([fake_df, true_df]).sample(frac=1).reset_index(drop=True)
        #
        # dataset = Dataset.from_pandas(df)
        dataset = load_dataset("HuggingFaceM4/ChartQA")
        return dataset.train_test_split(test_size=test_size, seed=200)

    # def format_prompt(self, info):
    #     ans = 'True' if info['label'] == 1 else 'False'
    #     prompt = (
    #         f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #         f"You are a news validator. Classify as REAL or FAKE.<|eot_id|>"
    #         f"<|start_header_id|>user<|end_header_id|>\n\n"
    #         f"Article: {info['text']}<|eot_id|>"
    #         f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    #         f"{ans}<|eot_id|>"
    #     )
    #     return {"prompt": prompt}

    def format_data(self, example):
        if example['image'].mode != 'RGB':
            example['image'] = example['image'].convert('RGB')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example['query']},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(example['label'])},
                ],
            },
        ]
        example["text"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return example


@dataclass
class LLaVADataCollator:
    processor: AutoProcessor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        @dataclass
        class LLaVADataCollator:
            processor: AutoProcessor

            def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
                texts = [f["text"] for f in features]
                images = [f["image"] for f in features]

                batch = self.processor( text           = texts,
                                        images         = images,
                                        return_tensors =" pt",
                                        padding        = True,
                                        truncation     = True,
                                        max_length     = 1024)
                labels = batch["input_ids"].clone()
                for i, text in enumerate(texts):
                    if "[/INST]" in text:
                        prompt_part = text.split("[/INST]")[0] + "[/INST]"
                        prompt_tokens = self.processor.tokenizer.encode(
                            prompt_part,
                            add_special_tokens=False
                        )
                        instruction_len = len(prompt_tokens)
                        labels[i, :instruction_len] = -100
                if self.processor.tokenizer.pad_token_id is not None:
                    labels[labels == self.processor.tokenizer.pad_token_id] = -100
                batch["labels"] = labels
                return batch

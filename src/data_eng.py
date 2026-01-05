import pandas as pd
from datasets import Dataset


class DataEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_and_split(self, fake_path, true_path, test_size=0.3):
        fake_df = pd.read_csv(fake_path).assign(label=0)
        true_df = pd.read_csv(true_path).assign(label=1)
        df = pd.concat([fake_df, true_df]).sample(frac=1).reset_index(drop=True)

        dataset = Dataset.from_pandas(df)
        return dataset.train_test_split(test_size=test_size, seed=200)

    def format_prompt(self, info):
        ans = 'True' if info['label'] == 1 else 'False'
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a news validator. Classify as REAL or FAKE.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Article: {info['text']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ans}<|eot_id|>"
        )
        return {"prompt": prompt}
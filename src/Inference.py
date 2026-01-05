import torch


class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, article_text, max_new_tokens=10):
        # Format the prompt exactly like the training data
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a news validator. Classify as REAL or FAKE.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Article: {article_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Extract only the assistant's new response
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.split("assistant")[-1].strip()
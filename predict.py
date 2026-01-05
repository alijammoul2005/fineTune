from config.model_config import MODEL_ID
from src.model_eng import ModelEngine
from src.Inference import Predictor
from peft import PeftModel


MODEL_PATH = "./final_merged_model"

print("--- Initializing Prediction Engine ---")


model, tokenizer = ModelEngine.build_qlora_model(MODEL_PATH, {"r": 16}) # Dummy cfg for loader
model.eval()

engine = Predictor(model, tokenizer)

sample_article = "Scientists discover that eating chocolate makes you live until 200."
result = engine.predict(sample_article)

print(f"\n[Input]: {sample_article}")
print(f"[Verdict]: {result}")
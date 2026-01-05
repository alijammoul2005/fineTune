
MODEL_ID = "meta-llama/Llama-3.2-3B"
HF_REPO_ID = "Ali-jammoul/fake-news-detector-3b"
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

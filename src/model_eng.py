from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


class ModelEngine:
    @staticmethod
    def build_qlora_model(model_id, lora_cfg):
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.gradient_checkpoint_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(**lora_cfg, bias="none", task_type="CAUSAL_LM")
        return get_peft_model(model, peft_config), tokenizer
from config.model_config import *
from config.train_config import TRAINING_ARGS
from src.model_eng import ModelEngine
from src.data_eng import DataEngine
from trl import SFTTrainer, SFTConfig


model, tokenizer = ModelEngine.build_qlora_model(MODEL_ID, {
    "r": LORA_R, "lora_alpha": LORA_ALPHA, "target_modules": TARGET_MODULES
})

engine = DataEngine(tokenizer)
data = engine.load_and_split("fake.csv", "true.csv", test_size=TRAINING_ARGS['test_size'])
data = data.map(engine.format_prompt)

sft_config = SFTConfig(dataset_text_field="prompt", **TRAINING_ARGS)
trainer = SFTTrainer(
    model=model,
    train_dataset=data['train'],
    eval_dataset=data['test'],
    args=sft_config
)
trainer.train()

model.save_pretrained("./final_adapter")
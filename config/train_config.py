

TRAINING_ARGS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-4,
    "bf16": True,
    "output_dir": "outputs",
    "test_size": 0.3
}
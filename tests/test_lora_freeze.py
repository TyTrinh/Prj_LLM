"""Unit test to verify LoRA injection and parameter freezing."""

import torch
from src.config import ModelConfig
from src.model.transformer import GPTModel
from src.peft.lora_manager import LoRAManager
from src.peft.lora_layer import LoRALinear


def test_lora_injection_and_freezing():
    cfg = ModelConfig(
        vocab_size=256,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ffn=128,
    )
    model = GPTModel(cfg)

    # Inject LoRA into q_proj and v_proj
    LoRAManager.inject_lora(model, r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    LoRAManager.freeze_base_model(model)

    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            assert "lora_" in name, f"Non-LoRA param {name} is not frozen!"
        else:
            assert "lora_" not in name, f"LoRA param {name} is mistakenly frozen!"

    trainable_ratio = trainable_params / total_params
    assert trainable_ratio < 0.10, f"Trainable ratio too high: {trainable_ratio:.2%}"

"""Unit test to verify GPTModel forward pass output shape [B, T, V]."""

import torch
from src.config import ModelConfig
from src.model.transformer import GPTModel


def test_gpt_model_forward_shape():
    cfg = ModelConfig(
        vocab_size=256,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ffn=128,
    )
    model = GPTModel(cfg)
    model.eval()

    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
        f"Expected shape {(batch_size, seq_len, cfg.vocab_size)}, got {logits.shape}"
    )

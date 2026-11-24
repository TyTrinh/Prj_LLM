"""Sanity check test to verify all subpackages can be imported cleanly."""

def test_imports():
    import src
    from src.config import ModelConfig, TrainConfig, LoRAConfig, DPOConfig, GuardrailConfig
    
    cfg = ModelConfig()
    assert cfg.d_model == 512
    assert cfg.d_ffn is not None
    assert cfg.vocab_size == 8192
    
    lora_cfg = LoRAConfig()
    assert lora_cfg.r == 8
    
    dpo_cfg = DPOConfig()
    assert dpo_cfg.beta == 0.1

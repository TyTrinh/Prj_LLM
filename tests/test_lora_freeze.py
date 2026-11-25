"""Unit test to verify LoRA implementation, injection, freezing, toggling, merge_and_unload, and save/load weights."""

import os
import tempfile
import torch
import torch.nn as nn
from src.config import ModelConfig
from src.model.transformer import GPTModel
from src.peft.lora_manager import LoRAManager
from src.peft.lora_layer import LoRALinear


def test_lora_math_and_forward():
    """Kiểm tra tính toán toán học: h = W_0 x + (alpha / r) * B A x"""
    torch.manual_seed(42)
    in_dim, out_dim, r = 16, 32, 4
    alpha = 8.0
    scaling = alpha / r

    base_layer = nn.Linear(in_dim, out_dim, bias=False)
    lora_layer = LoRALinear(base_layer, r=r, lora_alpha=alpha)

    x = torch.randn(2, 5, in_dim)

    # Khi B = 0 lúc khởi tạo, output của LoRA phải bằng đúng output của base_layer
    out_init = lora_layer(x)
    base_out = base_layer(x)
    assert torch.allclose(out_init, base_out, atol=1e-6), "Initial output with B=0 should equal base_layer output"

    # Gán B khác 0 để kiểm tra công thức forward
    with torch.no_grad():
        lora_layer.lora_B.copy_(torch.randn_like(lora_layer.lora_B))

    out_lora = lora_layer(x)
    expected_delta = (x @ lora_layer.lora_A.T @ lora_layer.lora_B.T) * scaling
    expected_out = base_layer(x) + expected_delta
    assert torch.allclose(out_lora, expected_out, atol=1e-5), "LoRA forward computation mismatch"


def test_lora_enable_toggle():
    """Kiểm tra tính năng bật/tắt LoRA (phục vụ DPO reference)."""
    torch.manual_seed(42)
    base_layer = nn.Linear(8, 8, bias=False)
    lora_layer = LoRALinear(base_layer, r=2, lora_alpha=4.0)

    with torch.no_grad():
        lora_layer.lora_B.copy_(torch.randn_like(lora_layer.lora_B))

    x = torch.randn(1, 8)
    out_enabled = lora_layer(x)
    
    # Tắt LoRA
    lora_layer.enabled = False
    out_disabled = lora_layer(x)
    base_out = base_layer(x)

    assert torch.allclose(out_disabled, base_out, atol=1e-6)
    assert not torch.allclose(out_enabled, out_disabled, atol=1e-4)


def test_lora_injection_and_freezing():
    """Kiểm tra inject vào GPTModel, freeze base model và tỷ lệ tham số trainable < 3%."""
    cfg = ModelConfig(
        vocab_size=1000,
        block_size=128,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ffn=256,
    )
    model = GPTModel(cfg)

    # Inject LoRA vào q_proj và v_proj
    LoRAManager.inject_lora(model, r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    LoRAManager.freeze_base_model(model)

    trainable_params, total_params, trainable_ratio = LoRAManager.get_trainable_parameters(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"Non-LoRA param {name} is not frozen!"
        else:
            assert "lora_" not in name, f"LoRA param {name} is mistakenly frozen!"

    assert trainable_ratio < 0.03, f"Trainable ratio too high: {trainable_ratio:.2%} (expected < 3%)"


def test_merge_and_unload_zero_latency():
    """Kiểm tra merge_and_unload gập trọng số và dỡ bỏ wrapper LoRA, output giữ nguyên tuyệt đối."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=256,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ffn=128,
    )
    model = GPTModel(cfg)
    LoRAManager.inject_lora(model, r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])

    # Khởi tạo giá trị khác 0 cho lora_B để delta_W != 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.copy_(torch.randn_like(module.lora_B) * 0.1)

    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        logits_before_merge = model(x)

    # Gập trọng số và gỡ bỏ LoRALinear wrapper
    LoRAManager.merge_and_unload(model)

    # Xác minh không còn LoRALinear trong model
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_layers) == 0, f"Expected 0 LoRALinear layers after unload, found {len(lora_layers)}"

    with torch.no_grad():
        logits_after_merge = model(x)

    # Output phải trùng khớp với sai số cực nhỏ
    assert torch.allclose(logits_before_merge, logits_after_merge, atol=1e-5), "Logits mismatch after merge_and_unload!"


def test_lora_save_and_load_weights():
    """Kiểm tra lưu và nạp riêng rẽ Adapter weights (~vài KB/MB) mà không cần base weights."""
    torch.manual_seed(1337)
    cfg = ModelConfig(
        vocab_size=256,
        block_size=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ffn=128,
    )
    # 1. Khởi tạo Base Model ban đầu
    base_model = GPTModel(cfg)

    # Khởi tạo model A từ base_model
    model_a = GPTModel(cfg)
    model_a.load_state_dict(base_model.state_dict())
    LoRAManager.inject_lora(model_a, r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])

    # Mô phỏng huấn luyện LoRA: cập nhật cả A và B
    with torch.no_grad():
        for module in model_a.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.copy_(torch.randn_like(module.lora_A))
                module.lora_B.copy_(torch.randn_like(module.lora_B) * 0.5)

    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        logits_a = model_a(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "adapter_test.pt")
        # 2. Lưu adapter
        LoRAManager.save_lora_weights(
            model_a,
            adapter_path,
            metadata={"r": 4, "alpha": 8.0, "epoch": 1},
        )

        assert os.path.exists(adapter_path), "Adapter file was not created!"
        file_size_kb = os.path.getsize(adapter_path) / 1024
        # Adapter chỉ chứa ma trận LoRA nên file rất nhẹ (< 50 KB cho mô hình test)
        assert file_size_kb < 50, f"Adapter file unexpectedly large: {file_size_kb:.2f} KB"

        # 3. Khởi tạo model B từ cùng base_model (chưa có weights LoRA của A)
        model_b = GPTModel(cfg)
        model_b.load_state_dict(base_model.state_dict())
        LoRAManager.inject_lora(model_b, r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])

        # Trước khi nạp adapter, logits của B phải khác A (vì A của model B được init ngẫu nhiên độc lập)
        with torch.no_grad():
            logits_b_before = model_b(x)
        assert not torch.allclose(logits_a, logits_b_before, atol=1e-4)

        # 4. Nạp adapter vào model B
        meta = LoRAManager.load_lora_weights(model_b, adapter_path)
        assert meta.get("r") == 4
        assert meta.get("alpha") == 8.0

        # Sau khi nạp adapter, logits của B phải trùng khớp tuyệt đối với A
        with torch.no_grad():
            logits_b_after = model_b(x)
        assert torch.allclose(logits_a, logits_b_after, atol=1e-6), "Logits mismatch after loading adapter weights!"

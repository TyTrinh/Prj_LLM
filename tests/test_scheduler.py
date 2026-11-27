import pytest
import torch
from src.utils.scheduler import get_wsd_lr, WSDScheduler


def test_wsd_phases_and_transitions():
    max_steps = 1000
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_ratio = 0.05  # 50 steps
    decay_ratio = 0.10   # 100 steps (900 -> 1000)

    # 1. Warmup: step 0 -> lr bắt đầu tăng
    lr_0 = get_wsd_lr(0, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio)
    assert 0.0 < lr_0 <= (max_lr / 50) + 1e-7

    # Warmup end (step 49)
    lr_warmup_end = get_wsd_lr(49, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio)
    assert abs(lr_warmup_end - max_lr) < 1e-5

    # 2. Stable phase: step 50 đến 899 -> giữ nguyên tuyệt đối max_lr
    for s in [50, 200, 500, 750, 899]:
        lr_stable = get_wsd_lr(s, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio)
        assert lr_stable == pytest.approx(max_lr), f"Failed at step {s}"

    # 3. Decay phase: step 900 -> 1000
    lr_decay_start = get_wsd_lr(900, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio)
    assert lr_decay_start == pytest.approx(max_lr)

    lr_decay_mid = get_wsd_lr(950, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio)
    assert min_lr < lr_decay_mid < max_lr

    lr_decay_end = get_wsd_lr(1000, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio)
    assert lr_decay_end == pytest.approx(min_lr)

    # Vượt quá max_steps -> giữ min_lr
    assert get_wsd_lr(1050, max_steps, max_lr, min_lr, warmup_ratio, decay_ratio) == min_lr


def test_wsd_decay_types():
    max_steps = 1000
    max_lr = 1e-3
    min_lr = 1e-4

    # So sánh cosine, linear và exponential decay ở giữa giai đoạn decay
    step_mid = 950
    lr_cos = get_wsd_lr(step_mid, max_steps, max_lr, min_lr, decay_type="cosine")
    lr_lin = get_wsd_lr(step_mid, max_steps, max_lr, min_lr, decay_type="linear")
    lr_exp = get_wsd_lr(step_mid, max_steps, max_lr, min_lr, decay_type="exponential")

    for lr in [lr_cos, lr_lin, lr_exp]:
        assert min_lr < lr < max_lr


def test_wsd_pytorch_scheduler():
    param = torch.nn.Parameter(torch.zeros(10))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    scheduler = WSDScheduler(
        optimizer,
        max_steps=100,
        warmup_ratio=0.05,
        decay_ratio=0.10,
        min_lr_ratio=0.1,
    )

    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Bước 10 (trong stable): lr = 1e-3
    assert lrs[10] == pytest.approx(1e-3)
    # Bước 50 (trong stable): lr = 1e-3
    assert lrs[50] == pytest.approx(1e-3)
    # Bước 95 (trong decay): giảm dần
    assert 1e-4 < lrs[95] < 1e-3

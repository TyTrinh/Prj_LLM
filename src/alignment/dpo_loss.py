r"""
==============================================================================
FILE: src/alignment/dpo_loss.py
MODULE: src.alignment.dpo_loss
RESPONSIBILITY:
    - Đơn trách nhiệm: Tính toán hàm mất mát Direct Preference Optimization (DPO).
    - Triển khai công thức:
      $\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$
    - Tích hợp [CẢI TIẾN 2]: Length Normalization để chống hiện tượng thiên vị câu trả lời dài (Length Bias).
    - Tính toán Implicit Reward Margin $(r_w - r_l)$ và Accuracy để log metrics.
INPUT:
    - policy_chosen_logps, policy_rejected_logps: torch.Tensor [B]
    - ref_chosen_logps, ref_rejected_logps: torch.Tensor [B]
    - beta: float
OUTPUT:
    - loss: torch.Tensor (vô hướng scalar)
    - chosen_rewards, rejected_rewards: torch.Tensor [B]
DEPENDENCY:
    - Chỉ phụ thuộc `torch`. Độc lập với kiến trúc mạng cụ thể.
==============================================================================
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Tính tổng hoặc trung bình log-probability CHỈ trên các token thuộc phần response."""
    # Logits: [B, T, V], Labels: [B, T], Mask: [B, T]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_token_logps = per_token_logps * response_mask

    if average_log_prob:
        # Cải tiến: Length Normalization
        token_count = response_mask.sum(dim=-1).clamp(min=1.0)
        return per_token_logps.sum(dim=-1) / token_count
    else:
        return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tính DPO loss theo công thức chuẩn và trích xuất Implicit Rewards."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards

"""Unit test to verify DPO Loss calculation on synthetic tensors."""

import torch
from src.alignment.dpo_loss import dpo_loss


def test_dpo_loss_computation():
    batch_size = 4
    # Case 1: Policy assigns higher log-prob to chosen than rejected (aligned model)
    policy_chosen_logps = torch.tensor([-1.0, -1.2, -0.8, -1.1])
    policy_rejected_logps = torch.tensor([-3.5, -4.0, -2.9, -3.8])

    ref_chosen_logps = torch.tensor([-2.0, -2.1, -2.0, -2.0])
    ref_rejected_logps = torch.tensor([-2.2, -2.3, -2.1, -2.1])

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.1,
    )

    assert not torch.isnan(loss), "DPO Loss returned NaN!"
    assert loss.ndim == 0, "DPO Loss should be a scalar tensor"
    assert (chosen_rewards > rejected_rewards).all(), "Chosen rewards should exceed rejected rewards"

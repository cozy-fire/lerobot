"""Tests for VLAFlowMatching.forward() training pass (§3.8).

VLAFlowMatching.forward() returns raw per-element MSE losses (B, chunk, action_dim).
Scalar loss = losses.mean().
"""

import pytest
import torch


@pytest.mark.slow
class TestForward:
    def test_forward_returns_losses_tensor(self, vla_flow_matching, batch):
        """forward() returns (B, chunk, action_dim) MSE tensor."""
        vfm = vla_flow_matching
        losses = vfm.forward(**batch)
        assert isinstance(losses, torch.Tensor)
        assert losses.ndim == 3  # (B, chunk, action_dim)

    def test_loss_scalar_mean(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        losses = vfm.forward(**batch)
        scalar = losses.mean()
        assert scalar.ndim == 0

    def test_loss_positive(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        losses = vfm.forward(**batch)
        assert losses.mean().item() >= 0

    def test_deterministic_same_input(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        vfm.eval()
        dev = batch["state"].device
        torch.manual_seed(42)
        noise1 = torch.randn(2, 50, 32, device=dev)
        torch.manual_seed(42)
        noise2 = torch.randn(2, 50, 32, device=dev)
        time_t = torch.tensor([0.5, 0.5], device=dev)
        l1 = vfm.forward(**batch, noise=noise1, time=time_t)
        l2 = vfm.forward(**batch, noise=noise2, time=time_t)
        assert torch.allclose(l1, l2)

    def test_different_noise_different_loss(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        dev = batch["state"].device
        l1 = vfm.forward(**batch, noise=torch.randn(2, 50, 32, device=dev))
        l2 = vfm.forward(**batch, noise=torch.randn(2, 50, 32, device=dev))
        assert l1.mean().item() != l2.mean().item()

    def test_highways_used(self, vla_flow_matching, batch):
        """With valid highways, loss should differ from zeroed highways."""
        vfm = vla_flow_matching
        l1 = vfm.forward(**batch)
        with torch.no_grad():
            for proj in vfm.vl_highway_proj:
                proj.weight.zero_()
        l2 = vfm.forward(**batch)
        assert l1.mean().item() != l2.mean().item()

    def test_stage0_vs_stage1_different_loss(self, vla_stage0, vla_stage1, batch):
        """Stage 0 and Stage 1 have different architectures -> different loss."""
        l0 = vla_stage0.forward(**batch)
        l1 = vla_stage1.forward(**batch)
        assert l0.mean().item() != l1.mean().item()

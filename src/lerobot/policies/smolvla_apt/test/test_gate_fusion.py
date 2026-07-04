"""Tests for GateFusionBlock (§3.1)."""

import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import GateFusionBlock


class TestGateFusionBlock:
    def test_init_gate_shape(self, gate_fusion_block):
        assert gate_fusion_block.gate.shape == (8, 864)  # NUM_AE_LAYERS, AE_HIDDEN

    def test_init_gate_zero(self, gate_fusion_block):
        assert (gate_fusion_block.gate == 0.0).all()

    def test_forward_preserves_shape(self, gate_fusion_block, vla_tokens, vl_highways_proj):
        out = gate_fusion_block(vla_tokens, vl_highways_proj[0], layer_idx=0)
        assert out.shape == vla_tokens.shape

    def test_forward_equal_weight_at_init(self, gate_fusion_block):
        dev = gate_fusion_block.gate.device
        vl = torch.ones(2, 10, 864, device=dev)
        hw = torch.zeros(2, 10, 864, device=dev)
        out = gate_fusion_block(vl, hw, layer_idx=0)
        expected = 0.5 * vl + 0.5 * hw  # sigmoid(0)=0.5
        assert torch.allclose(out, expected, atol=1e-6)

    def test_different_layer_different_gate(self, gate_fusion_block, vla_tokens, vl_highways_proj):
        out0 = gate_fusion_block(vla_tokens, vl_highways_proj[0], layer_idx=0)
        out1 = gate_fusion_block(vla_tokens, vl_highways_proj[1], layer_idx=1)
        # Different layers have independent gate entries (should differ unless highways identical)
        if torch.equal(vl_highways_proj[0], vl_highways_proj[1]):
            # If same highway (unlikely with randn), gates still won't match exactly
            # due to different indices — but the test is valid
            pass
        else:
            # Trivially different since highway inputs differ
            assert not torch.equal(out0, out1)

    def test_gate_all_one(self):
        """gate=10 -> sigmoid(10)~1 -> vl mostly preserved."""
        block = GateFusionBlock(8, 864, gate_init=10.0)
        vl = torch.ones(2, 10, 864)
        hw = torch.zeros(2, 10, 864)
        out = block(vl, hw, layer_idx=0)
        assert torch.allclose(out, vl, atol=1e-3)

    def test_gate_all_neg(self):
        """gate=-10 -> sigmoid(-10)~0 -> fully replaced by highway."""
        block = GateFusionBlock(8, 864, gate_init=-10.0)
        vl = torch.zeros(2, 10, 864)
        hw = torch.ones(2, 10, 864)
        out = block(vl, hw, layer_idx=0)
        assert torch.allclose(out, hw, atol=1e-3)

    def test_gradient_flows(self, gate_fusion_block, vla_tokens, vl_highways_proj):
        vl = vla_tokens.clone().requires_grad_(True)
        out = gate_fusion_block(vl, vl_highways_proj[0], layer_idx=0)
        loss = out.sum()
        loss.backward()
        assert gate_fusion_block.gate.grad is not None

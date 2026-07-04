"""Tests for AttentionBlock (§3.3)."""

import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import AttentionBlock


class TestAttentionBlock:
    def test_forward_shape(self, attention_block, vla_tokens, attn_mask_2d):
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        out = attention_block(vla_tokens, attention_mask=mask)
        assert out.shape == vla_tokens.shape

    def test_residual_connection(self, attention_block, vla_tokens):
        """When attention is zeroed via mask, output should not be NaN."""
        L = vla_tokens.shape[1]
        zero_mask = torch.zeros(2, L, L, dtype=torch.bool, device=vla_tokens.device)
        out = attention_block(vla_tokens, attention_mask=zero_mask)
        assert not torch.isnan(out).any()
        assert out.shape == vla_tokens.shape

    def test_swiglu_ffn(self, attention_block):
        """SwiGLU FFN has gate_proj, up_proj, down_proj."""
        assert hasattr(attention_block, "gate_proj")
        assert hasattr(attention_block, "up_proj")
        assert hasattr(attention_block, "down_proj")
        assert attention_block.gate_proj.bias is None

    def test_rms_norm_present(self, attention_block):
        """norm1 and norm2 are RMSNorm."""
        assert isinstance(attention_block.norm1, torch.nn.RMSNorm)
        assert isinstance(attention_block.norm2, torch.nn.RMSNorm)

    def test_no_film(self):
        """Verify AdaRMSNorm is NOT used (FiLM deleted per S2)."""
        from lerobot.policies.smolvla_apt import modeling_smolvla_apt
        assert not hasattr(modeling_smolvla_apt, "AdaRMSNorm")

    def test_rope_pe_type(self, vla_tokens, attn_mask_2d):
        block = AttentionBlock(864, 16, pe_type=(None, "rope")).to(vla_tokens.device)
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        out = block(vla_tokens, attention_mask=mask)
        assert out.shape == vla_tokens.shape

    def test_sinusoidal_pe_type(self, vla_tokens, attn_mask_2d):
        block = AttentionBlock(864, 16, pe_type=("sinusoidal", None)).to(vla_tokens.device)
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        out = block(vla_tokens, attention_mask=mask)
        assert out.shape == vla_tokens.shape

    def test_gradient_flows(self, attention_block, vla_tokens, attn_mask_2d):
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        x = vla_tokens.clone().requires_grad_(True)
        out = attention_block(x, attention_mask=mask)
        loss = out.sum()
        loss.backward()
        for name, param in attention_block.named_parameters():
            assert param.grad is not None, f"{name} should have grad"

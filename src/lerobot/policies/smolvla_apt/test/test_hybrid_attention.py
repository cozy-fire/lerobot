"""Tests for HybridAttentionLayers (§3.2)."""

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import HybridAttentionLayers


class TestHybridAttentionLayers:
    def test_stage0_num_layers(self, hybrid_attn_stage0):
        """Stage 0: ceil(8/2) = 4 layers."""
        assert hybrid_attn_stage0.num_active_layers == 4
        assert len(hybrid_attn_stage0.layers) == 4

    def test_stage1_num_layers(self, hybrid_attn_stage1):
        """Stage 1: full 8 layers."""
        assert hybrid_attn_stage1.num_active_layers == 8
        assert len(hybrid_attn_stage1.layers) == 8

    def test_stage1_even_num_layers(self):
        """Stage 1 requires even num_layers."""
        with pytest.raises(ValueError, match="even"):
            HybridAttentionLayers(864, 16, 7, train_stage=1)

    def test_forward_shape_preserved(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=None,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        assert out.shape == x.shape

    def test_gate_fusion_applied(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out_with = hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=None,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        out_without = hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=None,
            vla_split_sizes=(59, 51),
            vl_highways=[None] * 8,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        assert not torch.equal(out_with, out_without)

    def test_no_highway_no_crash(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=None,
            vla_split_sizes=(59, 51),
            vl_highways=[None] * 8,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        assert out.shape == x.shape

    def test_stage0_highway_matches_gate_index(
        self, hybrid_attn_stage0, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj, dilated_mask
    ):
        """A1 fix: Stage 0 uses highway[j] where j = gate_idx = 2*i+1, not highway[i]."""
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        accessed_indices = []
        original_fusion = hybrid_attn_stage0.gate_fusion.forward

        def tracking_fusion(vl, hw, layer_idx):
            accessed_indices.append(layer_idx)
            return original_fusion(vl, hw, layer_idx)

        hybrid_attn_stage0.gate_fusion.forward = tracking_fusion
        hybrid_attn_stage0(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=dilated_mask,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        hybrid_attn_stage0.gate_fusion.forward = original_fusion
        # Stage 0: 4 layers (i=0,1,2,3), gate_idx=j=2*i+1 -> [1,3,5,7]
        assert accessed_indices == [1, 3, 5, 7]

    def test_stage1_odd_layers_use_dilated(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj, dilated_mask
    ):
        """A2 fix: Stage 1 odd layers use dilated_mask, even layers use full_mask."""
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        masks_used = []

        # Patch each layer's forward to capture the attention_mask argument
        originals = []
        for layer in hybrid_attn_stage1.layers:
            orig_forward = layer.forward

            def make_patched(orig):
                def patched_forward(x_in, attention_mask, position_ids=None):
                    masks_used.append(attention_mask)
                    return orig(x_in, attention_mask=attention_mask, position_ids=position_ids)
                return patched_forward

            layer.forward = make_patched(orig_forward)
            originals.append(orig_forward)

        hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,     # full mask
            dilated_mask=dilated_mask,        # lang-blocked mask
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        # Restore originals
        for layer, orig in zip(hybrid_attn_stage1.layers, originals):
            layer.forward = orig

        assert len(masks_used) == 8
        # Odd layers (1,3,5,7) = dilated_mask; Even layers (0,2,4,6) = full_mask
        for i in [1, 3, 5, 7]:
            assert masks_used[i] is dilated_mask
        for i in [0, 2, 4, 6]:
            assert masks_used[i] is attn_mask_2d

    def test_stage0_all_va_pe(self, hybrid_attn_stage0):
        """Stage 0: all pe_types = ('sinusoidal', None)."""
        for layer in hybrid_attn_stage0.layers:
            assert layer.pe_type == ("sinusoidal", None)

    def test_stage1_interleaved_pe(self, hybrid_attn_stage1):
        """Stage 1: even=(None, 'rope'), odd=('sinusoidal', None)."""
        for i, layer in enumerate(hybrid_attn_stage1.layers):
            if i % 2 == 0:
                assert layer.pe_type == (None, "rope")
            else:
                assert layer.pe_type == ("sinusoidal", None)

    def test_vl_split_and_merge(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        """VL tokens (59) + AE tokens (51) are split, gated, and merged correctly."""
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=None,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110, device=vla_tokens.device).unsqueeze(0).expand(2, -1),
        )
        assert out.shape[1] == 110

    def test_gate_count_matches_layers(self, hybrid_attn_stage1):
        """Gate fusion has 8 gates (full Stage 1 count)."""
        assert hybrid_attn_stage1.gate_fusion.gate.shape[0] == 8

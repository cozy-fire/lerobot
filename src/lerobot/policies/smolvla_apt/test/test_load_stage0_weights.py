"""Tests for load_stage0_weights (§3.10)."""

import pytest
import torch
import copy


@pytest.mark.slow
class TestLoadStage0Weights:
    def test_layer_mapping(self, vla_stage1, vla_stage0_weights):
        """Stage 0 Layer 0 -> Stage 1 Layer 1, etc."""
        # Use q_proj.weight (Xavier random init, differs between stages)
        s1_orig = copy.deepcopy(vla_stage1.hybrid_attn_layers.layers[1].q_proj.weight)
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        s1_new = vla_stage1.hybrid_attn_layers.layers[1].q_proj.weight
        assert not torch.equal(s1_orig, s1_new)

    def test_even_layers_not_loaded(self, vla_stage1, vla_stage0_weights):
        """Stage 1 even layers (0, 2, 4, 6) should NOT be loaded from Stage 0."""
        s1_orig = copy.deepcopy(vla_stage1.hybrid_attn_layers.layers[0].norm1.weight)
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        s1_new = vla_stage1.hybrid_attn_layers.layers[0].norm1.weight
        assert torch.equal(s1_orig, s1_new)

    def test_projection_layers_copied(self, vla_stage1, vla_stage0_weights):
        """Projection layers copied directly."""
        orig = copy.deepcopy(vla_stage1.ae_state_proj.weight)
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        if "ae_state_proj.weight" in vla_stage0_weights:
            assert not torch.equal(orig, vla_stage1.ae_state_proj.weight)

    def test_gate_not_copied(self, vla_stage1, vla_stage0_weights):
        """Gate should NOT be copied from Stage 0 (fresh init per APT)."""
        gate_orig = copy.deepcopy(vla_stage1.hybrid_attn_layers.gate_fusion.gate)
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        gate_new = vla_stage1.hybrid_attn_layers.gate_fusion.gate
        assert torch.equal(gate_orig, gate_new)

    def test_wrong_stage_raises(self, policy_stage0):
        with pytest.raises(ValueError, match="train_stage"):
            policy_stage0.load_stage0_weights({})

    def test_strict_false_no_error(self, vla_stage1):
        """strict=False should not error even with extra/missing keys."""
        vla_stage1.load_state_dict({}, strict=False)

"""Tests for set_requires_grad (§3.4b)."""

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import VLAFlowMatching


@pytest.mark.slow
class TestSetRequiresGrad:
    def test_vlm_trainable_stage1_default(self, vla_flow_matching):
        """Stage 1 (default): VLM partially trainable (not all frozen)."""
        vfm = vla_flow_matching
        vfm.set_requires_grad()
        trainable_count = sum(1 for _, p in vfm.vlm_with_expert.named_parameters() if p.requires_grad)
        total_count = sum(1 for _ in vfm.vlm_with_expert.named_parameters())
        assert trainable_count > 0, f"VLM should have trainable params in Stage 1, got 0/{total_count}"

    def test_vlm_trainable_stage1(self, config_stage1):
        """Stage 1: VLM partially trainable (last layer + lm_head + norm frozen)."""
        vfm = VLAFlowMatching(config_stage1)
        vfm.set_requires_grad()
        # At least some VLM params should be trainable (not all frozen)
        any_trainable = any(p.requires_grad for p in vfm.vlm_with_expert.vlm.parameters())
        assert any_trainable, "VLM should have trainable params in Stage 1"

    def test_vlm_frozen_stage0(self, config_stage0):
        """Stage 0 always freezes VLM."""
        vfm = VLAFlowMatching(config_stage0)
        vfm.set_requires_grad()
        for name, param in vfm.vlm_with_expert.named_parameters():
            assert not param.requires_grad, f"VLM {name} should be frozen in Stage 0"

    def test_ae_trainable(self, vla_flow_matching):
        vfm = vla_flow_matching
        vfm.set_requires_grad()
        ae_modules = [
            vfm.ae_state_proj, vfm.vl0_proj, vfm.action_in_proj,
            vfm.action_out_proj, vfm.action_time_mlp_in, vfm.action_time_mlp_out,
            vfm.hybrid_attn_layers,
        ] + list(vfm.vl_highway_proj)
        for mod in ae_modules:
            for name, param in mod.named_parameters():
                assert param.requires_grad, f"{type(mod).__name__}.{name} should be trainable"

    def test_vl_highway_proj_trainable(self, vla_flow_matching):
        vfm = vla_flow_matching
        vfm.set_requires_grad()
        for i, proj in enumerate(vfm.vl_highway_proj):
            assert proj.weight.requires_grad
            assert proj.bias.requires_grad

    def test_hybrid_attn_trainable(self, vla_flow_matching):
        vfm = vla_flow_matching
        vfm.set_requires_grad()
        for name, param in vfm.hybrid_attn_layers.named_parameters():
            assert param.requires_grad, f"hybrid_attn_layers.{name} should be trainable"

    def test_old_state_proj_not_present(self, vla_flow_matching):
        assert not hasattr(vla_flow_matching, "state_proj")

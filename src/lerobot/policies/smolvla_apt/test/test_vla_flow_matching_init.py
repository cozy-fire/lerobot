"""Tests for VLAFlowMatching.__init__ (§3.4)."""

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import (
    VLAFlowMatching, GateFusionBlock, HybridAttentionLayers
)


@pytest.mark.slow
class TestVLAFlowMatchingInit:
    def test_all_modules_created(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        assert hasattr(model, "vlm_with_expert")
        assert hasattr(model, "vl0_proj")
        assert hasattr(model, "vl_highway_proj")
        assert hasattr(model, "ae_state_proj")
        assert hasattr(model, "action_in_proj")
        assert hasattr(model, "action_out_proj")
        assert hasattr(model, "action_time_mlp_in")
        assert hasattr(model, "action_time_mlp_out")
        assert hasattr(model, "hybrid_attn_layers")
        assert isinstance(model.hybrid_attn_layers.gate_fusion, GateFusionBlock)
        assert isinstance(model.hybrid_attn_layers, HybridAttentionLayers)

    def test_state_proj_removed(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        assert not hasattr(model, "state_proj")

    def test_ae_state_proj_exists(self, vla_flow_matching):
        model = vla_flow_matching
        assert isinstance(model.ae_state_proj, torch.nn.Linear)
        assert model.ae_state_proj.in_features == model.config.max_state_dim
        assert model.ae_state_proj.out_features == model.vlm_with_expert.expert_hidden_size

    def test_vl0_proj_dim(self, vla_flow_matching):
        model = vla_flow_matching
        vlm_hidden = model.vlm_with_expert.config.text_config.hidden_size
        ae_hidden = model.vlm_with_expert.expert_hidden_size
        assert model.vl0_proj.in_features == vlm_hidden
        assert model.vl0_proj.out_features == ae_hidden

    def test_vl_highway_proj_count(self, vla_flow_matching):
        model = vla_flow_matching
        vlm_hidden = model.vlm_with_expert.config.text_config.hidden_size
        ae_hidden = model.vlm_with_expert.expert_hidden_size
        num_highways = model.config.num_vlm_layers // model.config.vl_highway_interval
        assert len(model.vl_highway_proj) == num_highways
        for proj in model.vl_highway_proj:
            assert proj.in_features == vlm_hidden
            assert proj.out_features == ae_hidden

    def test_hybrid_attn_exists(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        assert model.hybrid_attn_layers is not None

    def test_num_ae_layers_equals_num_highways(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        expected = config_stage0.num_vlm_layers // config_stage0.vl_highway_interval  # 8
        assert model.hybrid_attn_layers.gate_fusion.gate.shape[0] == expected

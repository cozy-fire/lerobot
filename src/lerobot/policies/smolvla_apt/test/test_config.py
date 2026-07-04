"""Tests for configuration fields (§1.1, §1.2)."""

import pytest
from lerobot.policies.smolvla_apt.configuration_smolvla_apt import SmolVLAAptConfig


class TestNewFields:
    def test_new_fields_exist(self):
        cfg = SmolVLAAptConfig()
        assert cfg.train_stage == 0
        assert cfg.vl_highway_interval == 2
        assert cfg.gate_fusion_init == 0.0

    def test_deprecated_fields_removed(self):
        cfg = SmolVLAAptConfig()
        assert not hasattr(cfg, "attention_mode")
        assert not hasattr(cfg, "self_attn_every_n_layers")
        assert not hasattr(cfg, "train_state_proj")


class TestTrainStageValidation:
    def test_train_stage_valid(self):
        for stage in (0, 1):
            cfg = SmolVLAAptConfig()
            cfg.train_stage = stage
            cfg.__post_init__()  # should not raise

    def test_train_stage_invalid_raises(self):
        cfg = SmolVLAAptConfig()
        cfg.train_stage = 2
        with pytest.raises(ValueError, match="train_stage"):
            cfg.__post_init__()

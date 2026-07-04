"""Tests for SmolVLAAptPolicy init & PEFT deletion (§3.11, §3.12)."""

import pytest
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import SmolVLAAptPolicy, VLAFlowMatching


@pytest.mark.slow
class TestPolicyInit:
    def test_stage0_weights_not_loaded(self, config_stage0, monkeypatch):
        """Stage 0 should not try to load stage0 weights."""
        config_stage0.train_stage = 0
        config_stage0.load_stage_0_path = "/fake/path"
        policy = SmolVLAAptPolicy(config_stage0)
        assert isinstance(policy.model, VLAFlowMatching)

    def test_peft_methods_deleted(self, config_stage0):
        policy = SmolVLAAptPolicy(config_stage0)
        # PEFT methods removed from SmolVLAAptPolicy itself (not inherited from parent)
        assert "_get_default_peft_targets" not in type(policy).__dict__
        assert "_validate_peft_config" not in type(policy).__dict__

    def test_model_created(self, config_stage0):
        policy = SmolVLAAptPolicy(config_stage0)
        assert isinstance(policy.model, VLAFlowMatching)

    def test_reset_called(self, config_stage0):
        policy = SmolVLAAptPolicy(config_stage0)
        assert hasattr(policy, "_queues")
        assert "action" in policy._queues

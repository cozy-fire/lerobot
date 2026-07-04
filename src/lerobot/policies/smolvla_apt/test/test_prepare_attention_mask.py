"""Tests for prepare_attention_mask (§3.7)."""

import pytest
import torch


@pytest.mark.slow
class TestPrepareAttentionMask:
    def test_full_mask_shape(self, vla_flow_matching):
        vfm = vla_flow_matching
        L_VL = 59  # vision(49) + language(10)
        L_AE = 51
        vl_mask = torch.ones(2, L_VL, dtype=torch.bool)
        ae_mask = torch.ones(2, L_AE, dtype=torch.bool)
        full, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        assert full.shape == (2, 110, 110)
        # VL→VL: Vision+Language tokens can attend to each other
        assert full[:, :L_VL, :L_VL].all()
        # VL→AE: Vision+Language tokens CANNOT attend to State/Action tokens
        assert not full[:, :L_VL, L_VL:].any()
        # AE→all: State/Action tokens can attend to everyone
        assert full[:, L_VL:, :].all()

    def test_dilated_mask_blocks_language(self, vla_flow_matching):
        """Language tokens (positions 49..59) blocked from vision and action."""
        vfm = vla_flow_matching
        vl_mask = torch.ones(2, 59, dtype=torch.bool)
        ae_mask = torch.ones(2, 51, dtype=torch.bool)
        _, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        # Language at col 50 should NOT be visible to vision at col 0
        assert not dilated[0, 0, 50]    # vision->lang blocked
        assert not dilated[0, 50, 0]    # lang->vision blocked
        assert not dilated[0, 100, 50]  # action->lang blocked
        assert not dilated[0, 50, 100]  # lang->action blocked

    def test_modality_vision_is_1(self, vla_flow_matching):
        """Vision tokens at positions 0..49 should be blocked from language."""
        vfm = vla_flow_matching
        vl_mask = torch.ones(3, 59, dtype=torch.bool)
        ae_mask = torch.ones(3, 51, dtype=torch.bool)
        _, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        assert not dilated[:, 0, 50].any()

    def test_no_cross_modality_leak(self, vla_flow_matching):
        """Exhaustive check for language<->vision and language<->action."""
        vfm = vla_flow_matching
        nv, nl, na = 49, 10, 51
        L = nv + nl + na
        vl_mask = torch.ones(1, nv + nl, dtype=torch.bool)
        ae_mask = torch.ones(1, na, dtype=torch.bool)
        _, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, nv, train_stage=0)
        for v in range(nv):
            for l_idx in range(nv, nv + nl):
                assert not dilated[0, v, l_idx], f"vision->lang leak at {v}->{l_idx}"
                assert not dilated[0, l_idx, v], f"lang->vision leak at {l_idx}->{v}"
        for a in range(nv + nl, L):
            for l_idx in range(nv, nv + nl):
                assert not dilated[0, a, l_idx], f"action->lang leak at {a}->{l_idx}"
                assert not dilated[0, l_idx, a], f"lang->action leak at {l_idx}->{a}"

    def test_full_mask_vl_blocked_from_ae(self, vla_flow_matching):
        """VL tokens (Vision+Language) must not attend to AE tokens (State+Action).
        AE tokens can attend to VL tokens."""
        vfm = vla_flow_matching
        vl_mask = torch.ones(2, 59, dtype=torch.bool)
        ae_mask = torch.ones(2, 51, dtype=torch.bool)
        full, _ = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        # Vision→AE blocked
        assert not full[0, 0, 60]
        # Language→AE blocked
        assert not full[0, 50, 60]
        # AE→Vision allowed
        assert full[0, 60, 0]
        # AE→Language allowed
        assert full[0, 60, 50]

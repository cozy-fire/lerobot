"""Tests for embed_ae_tokens (§3.6)."""

import pytest
import torch


@pytest.mark.slow
class TestEmbedAETokens:
    def test_returns_three_values(self, vla_flow_matching, actions, state):
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5], device=actions.device)
        result = vfm.embed_ae_tokens(actions, time, state)
        assert len(result) == 3

    def test_state_first(self, vla_flow_matching, actions, state):
        """State token is at position 0, before action tokens."""
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5], device=actions.device)
        embs, _, _ = vfm.embed_ae_tokens(actions, time, state)
        # The first token should match the state projection
        state_emb = vfm.ae_state_proj(state).unsqueeze(1)
        assert torch.allclose(embs[:, 0:1, :], state_emb, atol=1e-5)

    def test_output_shape(self, vla_flow_matching, actions, state):
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5], device=actions.device)
        embs, pad_masks, att_masks = vfm.embed_ae_tokens(actions, time, state)
        ae_hidden = vfm.vlm_with_expert.expert_hidden_size
        assert embs.shape == (2, 1 + 50, ae_hidden)  # B, 1+chunk, ae_hidden_dim
        assert pad_masks.shape == (2, 51)
        assert att_masks.shape == (2, 51)

    def test_action_time_fusion(self, vla_flow_matching, actions, state):
        """Different timesteps should produce different action embeddings."""
        vfm = vla_flow_matching
        time0 = torch.tensor([0.1, 0.1], device=actions.device)
        time1 = torch.tensor([0.9, 0.9], device=actions.device)
        embs0, _, _ = vfm.embed_ae_tokens(actions, time0, state)
        embs1, _, _ = vfm.embed_ae_tokens(actions, time1, state)
        # Action tokens (positions 1:) should differ
        assert not torch.equal(embs0[:, 1:, :], embs1[:, 1:, :])

    def test_pad_masks_all_true(self, vla_flow_matching, actions, state):
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5], device=actions.device)
        _, pad_masks, _ = vfm.embed_ae_tokens(actions, time, state)
        assert pad_masks.all()

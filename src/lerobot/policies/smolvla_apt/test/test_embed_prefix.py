"""Tests for embed_prefix (§3.5)."""

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import create_sinusoidal_pos_embedding


@pytest.mark.slow
class TestEmbedPrefix:
    def test_no_state_in_prefix(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
        """State is NOT in prefix; it moved to AE."""
        vfm = vla_flow_matching
        embs, pad_masks, att_masks, nvt = vfm.embed_prefix(
            [images], [img_masks], lang_tokens, lang_masks
        )
        # prefix has vision + language only (no state token)
        assert embs.shape[1] == nvt + lang_tokens.shape[1]

    def test_returns_four_values(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
        vfm = vla_flow_matching
        result = vfm.embed_prefix([images], [img_masks], lang_tokens, lang_masks)
        assert len(result) == 4
        embs, pad_masks, att_masks, num_vision_tokens = result
        assert isinstance(embs, torch.Tensor)
        assert isinstance(pad_masks, torch.Tensor)
        assert isinstance(att_masks, torch.Tensor)
        assert isinstance(num_vision_tokens, int)

    def test_num_vision_tokens_correct(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
        """num_vision_tokens equals image embedding token count."""
        vfm = vla_flow_matching
        _, _, _, nvt = vfm.embed_prefix([images], [img_masks], lang_tokens, lang_masks)
        assert nvt > 0

    def test_output_shapes_match(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
        vfm = vla_flow_matching
        embs, pad_masks, att_masks, nvt = vfm.embed_prefix(
            [images], [img_masks], lang_tokens, lang_masks
        )
        L = embs.shape[1]
        assert pad_masks.shape == (2, L)
        assert att_masks.shape == (2, L)
        assert nvt < L  # vision < total (vision + language)


class TestSinusoidalPosEmbedding:
    def test_output_shape(self):
        time = torch.tensor([0.2, 0.8])
        emb = create_sinusoidal_pos_embedding(time, dimension=864, min_period=4e-3, max_period=4.0)
        assert emb.shape == (2, 864)
        assert not torch.isnan(emb).any()

    def test_different_time_different_embedding(self):
        time = torch.tensor([0.2, 0.8])
        emb = create_sinusoidal_pos_embedding(time, dimension=864, min_period=4e-3, max_period=4.0)
        assert not torch.equal(emb[0], emb[1])

    def test_odd_dimension_raises(self):
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding(torch.tensor([0.5]), dimension=3, min_period=1e-3, max_period=1.0)

    def test_2d_time_raises(self):
        with pytest.raises(ValueError, match="shape"):
            create_sinusoidal_pos_embedding(
                torch.tensor([[0.5]]), dimension=16, min_period=1e-3, max_period=1.0
            )

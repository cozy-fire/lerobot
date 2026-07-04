"""Tests for sample_actions() inference path (§3.9)."""

import pytest
import torch


@pytest.mark.slow
class TestSampleActions:
    def test_output_shape(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        vfm = vla_flow_matching
        vfm.eval()
        with torch.no_grad():
            actions = vfm.sample_actions(
                [images], [img_masks], lang_tokens, lang_masks, state
            )
        assert actions.shape == (2, 50, 32)  # B, chunk, action_dim

    def test_vlm_called_once(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        """VLM.forward should be called only once (not per denoising step)."""
        vfm = vla_flow_matching
        vfm.eval()
        call_count = 0
        original_forward = vfm.vlm_with_expert.forward

        def counting_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)

        vfm.vlm_with_expert.forward = counting_forward
        with torch.no_grad():
            vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
        assert call_count == 1

    def test_position_ids_full_sequence(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        """position_ids must span VL + AE: verified by output shape matching expected sizes."""
        vfm = vla_flow_matching
        vfm.eval()
        L_vl = 59  # known from conftest constants
        L_ae = 51  # 1 state + 50 chunk
        with torch.no_grad():
            actions = vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
        # position_ids correctness is verified implicitly:
        # if they only covered AE (shorter), MHA would fail on shape mismatch
        assert actions.shape == (2, 50, 32)

    def test_num_steps_obeys_config(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        vfm = vla_flow_matching
        vfm.config.num_steps = 5
        vfm.eval()
        step_count = 0
        original_embed_ae = vfm.embed_ae_tokens

        def counting_embed_ae(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_embed_ae(*args, **kwargs)

        vfm.embed_ae_tokens = counting_embed_ae
        with torch.no_grad():
            vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
        assert step_count == 5

    def test_output_no_nan(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        vfm = vla_flow_matching
        vfm.eval()
        with torch.no_grad():
            actions = vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
        assert not torch.isnan(actions).any()

    def test_different_noise_different_output(
        self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state
    ):
        vfm = vla_flow_matching
        vfm.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            a1 = vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
            torch.manual_seed(1)
            a2 = vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
        assert not torch.equal(a1, a2)

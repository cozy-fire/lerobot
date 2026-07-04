"""Tests for VLM highway extraction (§2.1-2.2).

Uses real attention_mask/position_ids generated the same way
as VLAFlowMatching.forward() — via make_att_2d_masks + cumsum.
Both Stage 0 and Stage 1 are tested (same VLM highway behavior).
"""

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import make_att_2d_masks


def _build_vlm_inputs(prefix_embedding):
    """Build real attention_mask and position_ids from embed_prefix output."""
    embs, pad_masks, att_masks, _ = prefix_embedding
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    return embs, att_2d_masks, position_ids


class TestVLMHighwayExtraction:
    """Tests that run against the default Stage 1 model."""

    def test_forward_returns_three_tuples(self, smolvlm_model, prefix_embedding):
        """forward() returns (outputs_embeds, past_key_values, collected_hidden_states)."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding)
        _, _, highways = smolvlm_model.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert isinstance(highways, list)
        assert len(highways) > 0

    def test_highway_count_matches_interval(self, smolvlm_model, prefix_embedding):
        """num_highways = num_vlm_layers // vl_highway_interval."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding)
        interval = getattr(smolvlm_model, 'vl_highway_interval', 2)
        num_layers = len(smolvlm_model.get_vlm_model().text_model.layers)
        expected = num_layers // interval

        _, _, highways = smolvlm_model.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert len(highways) == expected

    def test_highway_shape(self, smolvlm_model, prefix_embedding):
        """Each highway = (B, L_prefix, vlm_hidden_dim)."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding)
        B_shape, L_shape, D_shape = embs.shape

        _, _, highways = smolvlm_model.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        for hw in highways:
            assert hw.shape == (B_shape, L_shape, D_shape)

    def test_highway_gradient_stage1(self, smolvlm_model, prefix_embedding):
        """Stage 1 highways should preserve gradients (requires_grad=True)."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding)
        _, _, highways = smolvlm_model.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        for hw in highways:
            assert hw.requires_grad

    def test_different_interval(self, smolvlm_model, prefix_embedding):
        """vl_highway_interval=4 -> fewer highways."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding)
        smolvlm_model.vl_highway_interval = 4
        num_layers = len(smolvlm_model.get_vlm_model().text_model.layers)
        expected = num_layers // 4

        _, _, highways = smolvlm_model.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert len(highways) == expected

    def test_forward_still_returns_past_kv(self, smolvlm_model, prefix_embedding):
        """past_key_values should still be returned (interface compat)."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding)
        _, past_kv, _ = smolvlm_model.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert isinstance(past_kv, dict) or past_kv is None


class TestVLMHighwayStage0:
    """Same tests against Stage 0 model (VLM frozen, highway behavior identical)."""

    @pytest.fixture
    def smolvlm_stage0(self, vla_stage0):
        return vla_stage0.vlm_with_expert

    @pytest.fixture
    def prefix_embedding_stage0(self, vla_stage0, images, img_masks, lang_tokens, lang_masks):
        return vla_stage0.embed_prefix(
            [images], [img_masks], lang_tokens, lang_masks
        )

    def test_highway_count_stage0(self, smolvlm_stage0, prefix_embedding_stage0):
        """Stage 0: highway count same as Stage 1 (VLM identical)."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding_stage0)
        interval = getattr(smolvlm_stage0, 'vl_highway_interval', 2)
        num_layers = len(smolvlm_stage0.get_vlm_model().text_model.layers)
        expected = num_layers // interval

        _, _, highways = smolvlm_stage0.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert len(highways) == expected

    def test_highways_detached_stage0(self, smolvlm_stage0, prefix_embedding_stage0):
        """Stage 0: VLM always frozen."""
        embs, att_mask, pos_ids = _build_vlm_inputs(prefix_embedding_stage0)
        _, _, highways = smolvlm_stage0.forward(
            attention_mask=att_mask,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        for hw in highways:
            assert not hw.requires_grad


# =============================================================================
# Position-ID tests (§ SmolVLA prefix-LM style; § APT interleaved PE types)
# =============================================================================

class TestVlmPositionIDs:
    """VLM position_ids = cumsum(pad_masks) - 1 (SmolVLA prefix-LM style).

    Position IDs are computed identically in every training stage
    because the VLM backbone is the same SmolVLM2-500M.
    """

    def test_position_ids_contiguous_from_zero(self, prefix_embedding):
        """Valid tokens get positions 0, 1, 2, ... in order."""
        _, pad_masks, _, _ = prefix_embedding
        pos = torch.cumsum(pad_masks, dim=1) - 1
        for b in range(pad_masks.shape[0]):
            valid = pos[b][pad_masks[b]]
            expected = torch.arange(valid.numel(), device=pos.device, dtype=pos.dtype)
            assert torch.equal(valid, expected), (
                f"batch {b}: valid positions {valid.tolist()} != {expected.tolist()}"
            )

    def test_padding_repeats_last_valid_position(self, prefix_embedding):
        """Padding (pad_mask=False) repeats the last valid position.

        Repeating is safe because padding tokens are masked in attention.
        SmolVLA uses the same convention to keep RoPE computation simple.
        """
        _, pad_masks, _, _ = prefix_embedding
        pos = torch.cumsum(pad_masks, dim=1) - 1
        for b in range(pad_masks.shape[0]):
            pad_idx = (~pad_masks[b]).nonzero(as_tuple=True)[0]
            if len(pad_idx) == 0:
                continue
            for pi in pad_idx:
                if pi > 0:
                    assert pos[b, pi] == pos[b, pi - 1], (
                        f"padding at {pi} should repeat position from {pi - 1}"
                    )

    def test_position_ids_same_across_stages(
        self, vla_stage0, vla_stage1, images, img_masks, lang_tokens, lang_masks
    ):
        """VLM position_ids are identical regardless of train_stage."""
        _, pad0, _, _ = vla_stage0.embed_prefix([images], [img_masks], lang_tokens, lang_masks)
        _, pad1, _, _ = vla_stage1.embed_prefix([images], [img_masks], lang_tokens, lang_masks)
        pos0 = torch.cumsum(pad0, dim=1) - 1
        pos1 = torch.cumsum(pad1, dim=1) - 1
        assert torch.equal(pad0, pad1), "pad_masks must be identical across stages"
        assert torch.equal(pos0, pos1), "position_ids must be identical across stages"

    def test_position_ids_dtype_is_long(self, prefix_embedding):
        """position_ids dtype is int64 — required for RoPE indexing."""
        _, pad_masks, _, _ = prefix_embedding
        pos = torch.cumsum(pad_masks, dim=1) - 1
        assert pos.dtype == torch.int64

    def test_position_ids_shape_matches_pad_masks(self, prefix_embedding):
        """position_ids and pad_masks have the same (B, L) shape."""
        _, pad_masks, _, _ = prefix_embedding
        pos = torch.cumsum(pad_masks, dim=1) - 1
        assert pos.shape == pad_masks.shape


class TestPositionEncodingAcrossStages:
    """Combined VL+AE position_ids and position-encoding-type assignment.

    APT reference:
      Stage 0 – all AE layers use sinusoidal PE (vision-action only).
      Stage 1 – interleaved: even layers = RoPE (language injection),
                odd layers = sinusoidal PE (vision-action).
    """

    # ---------- helpers ----------

    @staticmethod
    def _build_ae_tokens(vla, state, actions):
        """Return AE pad_masks (always all-True) using the real embed_ae_tokens."""
        bsize = actions.shape[0]
        device = actions.device
        noise = vla.sample_noise(actions.shape, device)
        time = vla.sample_time(bsize, device)
        time_e = time[:, None, None]
        x_t = time_e * noise + (1 - time_e) * actions
        _, ae_pad, _ = vla.embed_ae_tokens(x_t, time, state)
        return ae_pad

    def _combined_position_ids(self, vla, images, img_masks, lang_tokens, lang_masks, state, actions):
        """Replicate the combined position_ids computation from VLAFlowMatching.forward()."""
        _, prefix_pad, _, _ = vla.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        ae_pad = self._build_ae_tokens(vla, state, actions)
        x_pad = torch.cat([prefix_pad, ae_pad], dim=1)
        return torch.cumsum(x_pad, dim=1) - 1, prefix_pad.shape[1]

    # ---------- combined position_ids ----------

    def test_combined_positions_monotonic(self, vla_flow_matching, batch):
        """Position IDs must be monotonically non-decreasing across VL+AE."""
        pos, _ = self._combined_position_ids(
            vla_flow_matching,
            batch["images"], batch["img_masks"],
            batch["lang_tokens"], batch["lang_masks"],
            batch["state"], batch["actions"],
        )
        diffs = pos[:, 1:] - pos[:, :-1]
        assert (diffs >= 0).all(), f"non-monotonic positions:\n{pos}"

    def test_ae_starts_after_vl(self, vla_flow_matching, batch):
        """First AE position = last VL position + 1 (continuous boundary)."""
        pos, L_vl = self._combined_position_ids(
            vla_flow_matching,
            batch["images"], batch["img_masks"],
            batch["lang_tokens"], batch["lang_masks"],
            batch["state"], batch["actions"],
        )
        for b in range(pos.shape[0]):
            vl_max = pos[b, :L_vl].max()
            ae_min = pos[b, L_vl:].min()
            assert ae_min == vl_max + 1, (
                f"batch {b}: AE min {ae_min.item()} != VL max+1 = {vl_max.item() + 1}"
            )

    def test_ae_positions_all_valid(self, vla_flow_matching, batch):
        """AE tokens are never padded → every position ≥ 0."""
        pos, L_vl = self._combined_position_ids(
            vla_flow_matching,
            batch["images"], batch["img_masks"],
            batch["lang_tokens"], batch["lang_masks"],
            batch["state"], batch["actions"],
        )
        ae_pos = pos[:, L_vl:]
        assert (ae_pos >= 0).all()

    def test_ae_positions_contiguous(self, vla_flow_matching, batch):
        """AE positions increment by exactly 1 (state then each action chunk)."""
        pos, L_vl = self._combined_position_ids(
            vla_flow_matching,
            batch["images"], batch["img_masks"],
            batch["lang_tokens"], batch["lang_masks"],
            batch["state"], batch["actions"],
        )
        ae_pos = pos[:, L_vl:]
        diffs = ae_pos[:, 1:] - ae_pos[:, :-1]
        assert (diffs == 1).all(), f"AE positions should be contiguous, got diffs {diffs.unique()}"

    def test_stage0_combined_positions(self, vla_stage0, batch):
        """Stage 0 combined position_ids: monotonic, boundary-correct."""
        pos, L_vl = self._combined_position_ids(
            vla_stage0,
            batch["images"], batch["img_masks"],
            batch["lang_tokens"], batch["lang_masks"],
            batch["state"], batch["actions"],
        )
        for b in range(pos.shape[0]):
            vl_max = pos[b, :L_vl].max()
            ae_min = pos[b, L_vl:].min()
            assert ae_min == vl_max + 1

    def test_stage1_combined_positions(self, vla_stage1, batch):
        """Stage 1 combined position_ids: monotonic, boundary-correct."""
        pos, L_vl = self._combined_position_ids(
            vla_stage1,
            batch["images"], batch["img_masks"],
            batch["lang_tokens"], batch["lang_masks"],
            batch["state"], batch["actions"],
        )
        for b in range(pos.shape[0]):
            vl_max = pos[b, :L_vl].max()
            ae_min = pos[b, L_vl:].min()
            assert ae_min == vl_max + 1

    # ---------- PE type per stage (APT-style interleaving) ----------

    def test_stage0_all_layers_sinusoidal(self, vla_stage0):
        """Stage 0: every AE layer = ('sinusoidal', None) — no RoPE, no language."""
        layers = vla_stage0.hybrid_attn_layers.layers
        assert len(layers) == 4, f"Stage 0 should have ceil(8/2)=4 layers, got {len(layers)}"
        for i, layer in enumerate(layers):
            assert layer.pe_type == ("sinusoidal", None), (
                f"layer {i}: expected ('sinusoidal', None), got {layer.pe_type}"
            )

    def test_stage1_interleaved_pe_types(self, vla_stage1):
        """Stage 1: even layers = RoPE, odd layers = sinusoidal PE."""
        layers = vla_stage1.hybrid_attn_layers.layers
        assert len(layers) == 8, f"Stage 1 should have 8 layers, got {len(layers)}"
        for i, layer in enumerate(layers):
            if i % 2 == 0:
                assert layer.pe_type == (None, "rope"), (
                    f"layer {i} (even): expected (None, 'rope'), got {layer.pe_type}"
                )
            else:
                assert layer.pe_type == ("sinusoidal", None), (
                    f"layer {i} (odd): expected ('sinusoidal', None), got {layer.pe_type}"
                )

    def test_stage1_counts(self, vla_stage1):
        """Stage 1: exactly N/2 RoPE and N/2 sinusoidal layers."""
        pe_types = [layer.pe_type for layer in vla_stage1.hybrid_attn_layers.layers]
        n_rope = sum(1 for pt in pe_types if pt == (None, "rope"))
        n_sin = sum(1 for pt in pe_types if pt == ("sinusoidal", None))
        assert n_rope == 4, f"expected 4 RoPE layers, got {n_rope}"
        assert n_sin == 4, f"expected 4 sinusoidal layers, got {n_sin}"

    # mask-dispatch per PE type (even=full, odd=dilated) is already covered
    # by test_stage1_odd_layers_use_dilated in test_hybrid_attention.py.

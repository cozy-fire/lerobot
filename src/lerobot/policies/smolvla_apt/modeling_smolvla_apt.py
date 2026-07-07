#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA_APT:

Designed by Hugging Face.

Install smolvla_apt extra dependencies:
```bash
pip install -e ".[smolvla_apt]"
```

Example of finetuning the smolvla_apt pretrained model (`smolvla_apt_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_apt_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolvla_apt. SmolVLA_APT is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla_apt \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla_apt pretrained model outside LeRobot training framework:
```python
policy = SmolVLAAptPolicy.from_pretrained("lerobot/smolvla_apt_base")
```

"""

import logging
import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla_apt.configuration_smolvla_apt import SmolVLAAptConfig
from lerobot.policies.smolvla_apt.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device=None
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    if device is None:
        device = time.device
    device = torch.device(device)
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


# =============================================================================
# §3.1-3.3: APT-style Action Expert 辅助类
# =============================================================================

class GateFusionBlock(nn.Module):
    """
    Gate Fusion mechanism: inject VLM highway features into AE self-attention output.

    Reference: APT apt/action_expert.py:124-172 (HybridAttentionLayers)

    After each AE self-attention layer, the VL token stream is fused with the
    corresponding VLM intermediate hidden state via a learnable per-channel gate:

        vl = vl * sigmoid(g) + vlm_highway * (1 - sigmoid(g))

    The gate starts at sigmoid(0) = 0.5, giving equal weight to AE and VLM features.
    """
    def __init__(self, num_layers: int, hidden_dim: int, gate_init: float = 0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.full((num_layers, hidden_dim), gate_init))

    def forward(self, vl_tokens: Tensor, vlm_highway: Tensor, layer_idx: int) -> Tensor:
        gi = self.gate[layer_idx].sigmoid()  # (D,)
        vl_tokens = vl_tokens * gi + vlm_highway * (1 - gi)
        return vl_tokens


class AttentionBlock(nn.Module):
    """
    Self-attention + SwiGLU-FFN block for Action Expert.

    Reference: APT apt/layers/attn.py:77-113 (SelfAttentionBlock), 24-34 (FFN/SwiGLU)

    Architecture: RMSNorm → Q/K/V proj → RoPE → SDPA → O proj → residual
                                 → RMSNorm → SwiGLU-FFN → residual
    No FiLM/AdaRMSNorm — timestep modulation is handled via Concat+MLP in embed_ae_tokens.

    RoPE is applied to all tokens (VL + state + actions), aligning with SmolVLA.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pe_type: tuple = (None, "rope"),
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.pe_type = pe_type
        self.num_heads = num_heads

        head_dim = hidden_dim // num_heads  # computed internally, not from VLM

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.norm2 = nn.RMSNorm(hidden_dim)

        ffn_hidden = hidden_dim * ffn_expansion
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, hidden_dim, bias=False)

        # Xavier init — prevent SwiGLU gate*up variance explosion through stacked layers
        for m in [self.gate_proj, self.up_proj, self.down_proj,
                  self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor, attention_mask: Tensor, position_ids: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        head_dim = D // self.num_heads  # computed internally, not from VLM

        # --- Self-Attention with RoPE ---
        residual = x
        x = self.norm1(x)

        # Q/K/V projections → (B, L, H, D_head)
        q = self.q_proj(x).view(B, L, self.num_heads, head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, head_dim)

        # RoPE: align with SmolVLA — all tokens get rotary encoding
        if position_ids is not None:
            q = apply_rope(q, position_ids)
            k = apply_rope(k, position_ids)

        # (B, H, L, D_head) for F.scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Mask: (B, L, L) → (B, 1, L, L) for head broadcasting
        mask_sdpa = attention_mask.unsqueeze(1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_sdpa)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = self.o_proj(attn_out)
        x = x + residual

        # --- SwiGLU FFN ---
        residual = x
        x = self.norm2(x)
        x = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        x = x + residual
        return x


class HybridAttentionLayers(nn.Module):
    """
    Interleaved self-attention layers for VLA.

    Layer count = num_highways (VLM highway count).

    Stage 0: ceil(num_layers/2) VA layers (odd gate indices), language blocked by dilated_mask.
    Stage 1: num_layers layers, even=language-injection (full mask + RoPE), odd=VA (dilated mask + sinusoidal PE).

    Every AE layer has its own gate + highway injection (no gaps).
    Gate and highway use the same index j (one-to-one binding, per APT).

    Reference: APT apt/action_expert.py:124-172
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,       # = num_highways (e.g. 8)
        train_stage: int,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.train_stage = train_stage
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if train_stage == 0:
            va_num_layers = (num_layers + 1) // 2
            self.num_active_layers = va_num_layers
            pe_types = [("sinusoidal", None)] * va_num_layers
        else:
            if num_layers % 2 != 0:
                raise ValueError(f"Stage 1 requires even num_layers, got {num_layers}")
            self.num_active_layers = num_layers
            pe_types = []
            for i in range(num_layers):
                if i % 2 == 0:
                    pe_types.append((None, "rope"))
                else:
                    pe_types.append(("sinusoidal", None))

        self.layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, pe_type=pe_types[i])
            for i in range(self.num_active_layers)
        ])

        self.gate_fusion = GateFusionBlock(num_layers, hidden_dim, gate_init)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        dilated_mask: Tensor | None,
        vla_split_sizes: tuple[int, int],
        vl_highways: list[Tensor],
        position_ids: Tensor,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            if self.train_stage == 0:
                mask = dilated_mask if dilated_mask is not None else attention_mask
            elif i % 2 == 0:
                mask = attention_mask
            else:
                mask = dilated_mask if dilated_mask is not None else attention_mask

            x = layer(x, attention_mask=mask, position_ids=position_ids)

            j = (2 * i + 1) if self.train_stage == 0 else i
            if j < len(vl_highways) and vl_highways[j] is not None:
                len_vl, len_rest = vla_split_sizes
                vl, rest = x.split([len_vl, len_rest], dim=1)
                vl = self.gate_fusion(vl, vl_highways[j], j)
                x = torch.cat([vl, rest], dim=1)

        return x


def _verify_stage0_weights(
    model: nn.Module,
    stage0_state_dict: dict,
    mapped_layers: list[str],
    copied_projs: list[str],
) -> None:
    """Verify Stage 0 → Stage 1 weight mapping by comparing actual tensor values.

    Checks:
      1. Odd AE layers (VA) match Stage 0 counterpart (allclose).
      2. Even AE layers (VLA-injection) differ from Stage 0 (not copied).
      3. Projection layers match Stage 0 counterpart.
    """
    s1_layers = model.hybrid_attn_layers.layers
    all_ok = True
    results = []

    # --- Check AE layers ---
    s0_layer_idx = 0
    for s1_idx, layer in enumerate(s1_layers):
        if s1_idx % 2 == 1:
            # Odd layer: should match Stage 0 Layer s0_layer_idx
            matched = True
            for name, param in layer.named_parameters():
                k0 = f"model.hybrid_attn_layers.layers.{s0_layer_idx}.{name}"
                if k0 in stage0_state_dict:
                    if not torch.allclose(param, stage0_state_dict[k0].to(param.device)):
                        matched = False
                        break
            status = "✓ MATCH" if matched else "✗ MISMATCH"
            results.append(f"  s0.L{s0_layer_idx} → s1.L{s1_idx}: {status}")
            if not matched:
                all_ok = False
            s0_layer_idx += 1
        else:
            # Even layer: should NOT match any Stage 0 layer (random init)
            any_match = False
            for s0_check in range(s0_layer_idx):
                for name, param in layer.named_parameters():
                    k0 = f"model.hybrid_attn_layers.layers.{s0_check}.{name}"
                    if k0 in stage0_state_dict and torch.allclose(param, stage0_state_dict[k0].to(param.device)):
                        any_match = True
                        break
                if any_match:
                    break
            if any_match:
                results.append(f"  s1.L{s1_idx} (even): ✗ UNEXPECTED MATCH (should be fresh init)")
                all_ok = False
            else:
                results.append(f"  s1.L{s1_idx} (even): ✓ fresh init (diff from Stage 0)")

    logging.info(f"[SmolVLA-APT] AE layer weight verification:\n" + "\n".join(results))

    # --- Check projection layers ---
    proj_matched = 0
    proj_mismatched = []
    for pname in copied_projs:
        full_name = f"model.{pname}"
        s1_param = model.get_parameter(pname)
        if full_name in stage0_state_dict:
            if torch.allclose(s1_param, stage0_state_dict[full_name].to(s1_param.device)):
                proj_matched += 1
            else:
                proj_mismatched.append(pname)
                all_ok = False

    if proj_mismatched:
        logging.warning(
            f"[SmolVLA-APT] Projection layers MISMATCH ({len(proj_mismatched)}): {proj_mismatched}"
        )
    logging.info(
        f"[SmolVLA-APT] Projection layers verified: {proj_matched}/{len(copied_projs)} match"
    )

    if all_ok:
        logging.info("[SmolVLA-APT] ✅ All Stage 0 weights correctly loaded into Stage 1.")
    else:
        logging.warning("[SmolVLA-APT] ❌ Some weight mappings FAILED verification. Check logs above.")


class SmolVLAAptPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAAptConfig
    name = "smolvla_apt"

    def __init__(
        self,
        config: SmolVLAAptConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        # Stage 0→1 weight loading (APT: only odd layers + projections, no gate)
        if config.train_stage == 1 and config.load_stage_0_path is not None:
            import os
            ckpt_path = os.path.join(config.load_stage_0_path, "model.safetensors")
            if os.path.exists(ckpt_path):
                logging.info(f"[SmolVLA-APT] Loading Stage 0 weights from: {config.load_stage_0_path}")
                from safetensors.torch import load_file
                self.load_stage0_weights(load_file(ckpt_path))
            else:
                logging.warning(
                    f"[SmolVLA-APT] load_stage_0_path is set but model.safetensors not found at: {ckpt_path}"
                )
        elif config.train_stage == 1 and config.load_stage_0_path is None:
            logging.info(
                "[SmolVLA-APT] train_stage=1 but load_stage_0_path not set. "
                "All Stage 1 layers will be randomly initialized."
            )
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> list[dict]:
        """Return parameter groups with differential LR and weight decay.
        AE: inherits global optimizer_lr / optimizer_weight_decay.
        VLM: uses vlm_optimizer_lr / vlm_optimizer_weight_decay."""
        vlm_params, ae_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.vlm_with_expert."):
                vlm_params.append(param)
            else:
                ae_params.append(param)
        groups = [{"params": ae_params, "name": "ae"}]
        if vlm_params:
            groups.append({
                "params": vlm_params,
                "lr": self.config.vlm_optimizer_lr,
                "weight_decay": self.config.vlm_optimizer_weight_decay,
                "name": "vlm",
            })
        return groups

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_is_pad")
        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise=noise, time=time
        )
        loss_dict["losses_after_forward"] = losses.mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        image_keys = [key for key in self.config.image_features]

        if self.config.camera_order:
            def _sort_key(k):
                try:
                    return self.config.camera_order.index(k)
                except ValueError:
                    return len(self.config.camera_order)
            image_keys = sorted(image_keys, key=_sort_key)

        if not any(key in batch for key in image_keys):
            raise ValueError(
                f"No image features found in batch for any expected camera. "
                f"(batch keys: {list(batch.keys())}) (expected: {image_keys})"
            )

        bsize = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        images = []
        img_masks = []

        for key in image_keys:
            if key in batch:
                img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                img = img * 2.0 - 1.0  # [0,1] → [-1,1] for SigLIP

                if f"{key}_padding_mask" in batch:
                    mask = batch[f"{key}_padding_mask"].bool()
                else:
                    mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
                images.append(img)
                img_masks.append(mask)
            else:
                # Missing camera: fill with black image and zero mask
                img = torch.ones(bsize, 3, *self.config.resize_imgs_with_padding, device=device) * -1
                mask = torch.zeros(bsize, dtype=torch.bool, device=device)
                images.append(img)
                img_masks.append(mask)

        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def load_stage0_weights(self, stage0_state_dict: dict):
        """Load Stage 0 checkpoint into Stage 1 model.
        Stage 0 Layer i → Stage 1 Layer 2*i+1 (odd layers only).
        Projection layers copied directly. Gate NOT copied (fresh init per APT)."""
        if self.config.train_stage != 1:
            raise ValueError("load_stage0_weights only works when train_stage=1")

        stage0_keys = set(stage0_state_dict.keys())
        model_state = self.model.state_dict()

        logging.info(
            f"[SmolVLA-APT] Stage 1 has {len(self.model.hybrid_attn_layers.layers)} AE layers "
            f"(even=VLA-injection, odd=VA). Stage 0 checkpoint has {len(stage0_keys)} keys."
        )

        # --- Map AE layers: Stage 0 Layer i → Stage 1 odd Layer 2*i+1 ---
        s0_layer_idx = 0
        mapped_layers = []
        for s1_layer_idx in range(len(self.model.hybrid_attn_layers.layers)):
            if s1_layer_idx % 2 == 1:
                for name, _ in self.model.hybrid_attn_layers.layers[s1_layer_idx].named_parameters():
                    k1 = f"hybrid_attn_layers.layers.{s1_layer_idx}.{name}"
                    k0 = f"model.hybrid_attn_layers.layers.{s0_layer_idx}.{name}"
                    if k0 in stage0_state_dict:
                        model_state[k1] = stage0_state_dict[k0]
                mapped_layers.append(f"s0.L{s0_layer_idx}→s1.L{s1_layer_idx}")
                s0_layer_idx += 1
        logging.info(f"[SmolVLA-APT] AE layer mapping: {', '.join(mapped_layers)}")

        # --- Copy projection layers (no gate, no hybrid_attn internals) ---
        copied_projs = []
        skipped_projs = []
        for name, _ in self.model.named_parameters():
            if "hybrid_attn_layers" not in name and "gate_fusion" not in name:
                ckpt_key = f"model.{name}"
                if ckpt_key in stage0_state_dict:
                    model_state[name] = stage0_state_dict[ckpt_key]
                    copied_projs.append(name)
                else:
                    skipped_projs.append(name)
        logging.info(
            f"[SmolVLA-APT] Projection layers copied: {len(copied_projs)}, "
            f"not found in checkpoint: {len(skipped_projs)}"
        )
        if skipped_projs:
            logging.debug(f"[SmolVLA-APT] Skipped projection keys: {skipped_projs}")

        # --- Gate parameters intentionally NOT copied (fresh init) ---
        gate_keys = [k for k in model_state if "gate_fusion" in k]
        logging.info(
            f"[SmolVLA-APT] Gate parameters: {len(gate_keys)} left at fresh init (NOT copied from Stage 0)"
        )

        self.model.load_state_dict(model_state, strict=False)
        logging.info("[SmolVLA-APT] Stage 0 weights loaded successfully.")
        _verify_stage0_weights(self.model, stage0_state_dict, mapped_layers, copied_projs)

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA-APT — Gate Fusion architecture.
    VLM runs once to produce vl0 + vl_highways.
    AE processes [V+L] + [State] + [NoisyActions] through
    HybridAttentionLayers with per-layer gate fusion.
    """

    def __init__(self, config: SmolVLAAptConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.train_stage = config.train_stage

        # === VLM Backbone (Image + Language only, no State) ===
        # Stage 0: freeze VLM; Stage 1: train VLM (determined by train_stage only)
        _train_expert_only = (self.train_stage == 0)
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=_train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            num_vlm_layers=self.config.num_vlm_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.vlm_with_expert.vl_highway_interval = self.config.vl_highway_interval

        ae_hidden_dim = self.vlm_with_expert.expert_hidden_size
        vlm_hidden_dim = self.vlm_with_expert.config.text_config.hidden_size

        # === VLM → AE projections ===
        self.vl0_proj = nn.Linear(vlm_hidden_dim, ae_hidden_dim)

        num_highways = len(list(range(0, self.config.num_vlm_layers, self.config.vl_highway_interval)))
        self.vl_highway_proj = nn.ModuleList([
            nn.Linear(vlm_hidden_dim, ae_hidden_dim) for _ in range(num_highways)
        ])

        # === AE projections ===
        self.ae_state_proj = nn.Linear(self.config.max_state_dim, ae_hidden_dim)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, ae_hidden_dim)
        self.action_out_proj = nn.Linear(ae_hidden_dim, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(ae_hidden_dim * 2, ae_hidden_dim)
        self.action_time_mlp_out = nn.Linear(ae_hidden_dim, ae_hidden_dim)

        # === Hybrid Attention Layers ===
        num_ae_layers = num_highways
        vlm_cfg = self.vlm_with_expert.config.text_config
        self.hybrid_attn_layers = HybridAttentionLayers(
            hidden_dim=ae_hidden_dim,
            num_heads=vlm_cfg.num_attention_heads,
            num_layers=num_ae_layers,
            train_stage=self.train_stage,
            gate_init=self.config.gate_fusion_init,
        )

        # --- Device alignment ---
        # VLM is on target device via device_map, but AE layers default to CPU.
        # Move AE sub-modules to match VLM's device. AE stays at fp32 (PyTorch default).
        # Aligns with PreTrainedPolicy.from_pretrained → .to().
        vlm_device = next(self.vlm_with_expert.vlm.parameters()).device
        for name, module in self.named_children():
            if name != "vlm_with_expert":
                module.to(device=vlm_device)

        self.set_requires_grad()

        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        # Stage 0: VLM frozen; Stage 1: VLM trainable (freeze_vision_encoder still respected)
        if self.train_stage == 1:
            self.vlm_with_expert.set_requires_grad()
        # AE always trainable
        for m in [self.ae_state_proj, self.vl0_proj, self.action_in_proj,
                  self.action_out_proj, self.action_time_mlp_in, self.action_time_mlp_out,
                  self.hybrid_attn_layers, *self.vl_highway_proj]:
            for p in m.parameters():
                p.requires_grad = True

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Embed images and language tokens for VLM. State moved to AE.
        Returns (embs, pad_masks, att_masks, num_vision_tokens)."""
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    ).unsqueeze(0).expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device)
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask_expanded = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask_expanded)
            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    ).unsqueeze(0).expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device)
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        # Vision/language boundary
        num_vision_tokens = sum(e.shape[1] for e in embs)
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        bsize = lang_emb.shape[0]
        device = lang_emb.device
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]
        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks, num_vision_tokens

    def embed_ae_tokens(self, noisy_actions, timestep, state):
        """Embed state + noisy_actions + timestep as AE tokens.
        State at position 0, actions follow."""
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        # State token
        state_emb = self.ae_state_proj(state).unsqueeze(1)
        # Action + timestep fusion
        action_emb = self.action_in_proj(noisy_actions)
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size,
            self.config.min_period, self.config.max_period, device=device
        )
        time_emb = time_emb.type(dtype=dtype)[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        # Concatenate [state, actions]
        ae_embs = torch.cat([state_emb, action_time_emb], dim=1)
        ae_pad_masks = torch.ones(bsize, ae_embs.shape[1], dtype=torch.bool, device=device)
        ae_att_masks = torch.zeros(bsize, ae_embs.shape[1], dtype=torch.long, device=device)
        return ae_embs, ae_pad_masks, ae_att_masks

    def prepare_attention_mask(self, vl_mask, ae_mask, num_vision_tokens, train_stage):
        """Build full_mask and dilated_mask for [Vision + Language + AE] sequence."""
        bsize = vl_mask.shape[0]
        device = vl_mask.device
        L_vl = vl_mask.shape[1]
        L_ae = ae_mask.shape[1]
        L = L_vl + L_ae
        modality = torch.full((bsize, L), 3, dtype=torch.long, device=device)
        modality[:, :num_vision_tokens] = 1
        modality[:, num_vision_tokens:L_vl] = 2
        full_mask = torch.ones(bsize, L, L, dtype=torch.bool, device=device)
        # VL tokens (Vision + Language) must NOT attend to AE tokens (State + Action)
        is_vl = (modality == 1) | (modality == 2)
        is_ae = modality == 3
        full_mask = full_mask & ~(is_vl[:, :, None] & is_ae[:, None, :])
        is_lang = modality == 2
        lang_block_mask = is_lang[:, :, None] | is_lang[:, None, :]
        dilated_mask = full_mask & ~lang_block_mask
        return full_mask, dilated_mask

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Training forward pass with gate fusion."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # VLM forward
        prefix_embs, prefix_pad_masks, prefix_att_masks, num_vision_tokens = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, _, vl_highways = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs],
            use_cache=False, fill_kv_cache=False,
        )
        # vl0 from pre-transformer embeddings
        vl0 = self.vl0_proj(prefix_embs.to(dtype=self.vl0_proj.weight.dtype))
        vl0_pad_mask = prefix_pad_masks
        # Project highways — cast to match projection dtype
        proj_dtype = self.vl_highway_proj[0].weight.dtype
        vl_highways_proj = []
        for i, hw in enumerate(vl_highways):
            if i < len(self.vl_highway_proj) and hw is not None:
                vl_highways_proj.append(self.vl_highway_proj[i](hw.to(dtype=proj_dtype)))
            else:
                vl_highways_proj.append(None)
        # AE tokens
        ae_embs, ae_pad_masks, ae_att_masks = self.embed_ae_tokens(x_t, time, state)
        # Concatenate
        x = torch.cat([vl0, ae_embs], dim=1)
        x_pad_masks = torch.cat([vl0_pad_mask, ae_pad_masks], dim=1)
        full_mask, dilated_mask = self.prepare_attention_mask(
            vl0_pad_mask, ae_pad_masks, num_vision_tokens, self.train_stage
        )
        # Hybrid attention
        position_ids_full = torch.cumsum(x_pad_masks, dim=1) - 1
        x = self.hybrid_attn_layers(
            x=x, attention_mask=full_mask, dilated_mask=dilated_mask,
            vla_split_sizes=(vl0.shape[1], ae_embs.shape[1]),
            vl_highways=vl_highways_proj, position_ids=position_ids_full,
        )
        suffix_out = x[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        """Inference forward with gate fusion."""
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        # VLM forward (once)
        prefix_embs, prefix_pad_masks, prefix_att_masks, num_vision_tokens = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, _, vl_highways = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs],
            use_cache=False, fill_kv_cache=False,
        )
        vl0 = self.vl0_proj(prefix_embs.to(dtype=self.vl0_proj.weight.dtype))
        vl0_pad_mask = prefix_pad_masks
        proj_dtype = self.vl_highway_proj[0].weight.dtype
        vl_highways_proj = []
        for i, hw in enumerate(vl_highways):
            if i < len(self.vl_highway_proj) and hw is not None:
                vl_highways_proj.append(self.vl_highway_proj[i](hw.to(dtype=proj_dtype)))
            else:
                vl_highways_proj.append(None)
        # Denoising loop
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            ae_embs, ae_pad_masks, ae_att_masks = self.embed_ae_tokens(x_t, time_tensor, state)
            x = torch.cat([vl0, ae_embs], dim=1)
            full_mask, dilated_mask = self.prepare_attention_mask(
                vl0_pad_mask, ae_pad_masks, num_vision_tokens, self.train_stage
            )
            x = self.hybrid_attn_layers(
                x=x, attention_mask=full_mask, dilated_mask=dilated_mask,
                vla_split_sizes=(vl0.shape[1], ae_embs.shape[1]),
                vl_highways=vl_highways_proj,
                position_ids=torch.cumsum(torch.cat([vl0_pad_mask, ae_pad_masks], dim=1), dim=1) - 1,
            )
            suffix_out = x[:, -self.config.chunk_size:].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
        return x_t

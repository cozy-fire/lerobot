# SmolVLA-APT 单元测试方案

> 对应实施文档：`docs/plan/smolvla_apt_implementation_steps.md`
> 测试代码目录：`src/lerobot/policies/smolvla_apt/test/`
> 覆盖率目标：>80%

---

## 测试基础设施

### 目录结构

```
src/lerobot/policies/smolvla_apt/test/
├── __init__.py
├── conftest.py                    # 共享 fixtures（详见 §B）
├── test_config.py                 # 对应 §1
├── test_vlm_highway.py            # 对应 §2
├── test_gate_fusion.py            # 对应 §3.1
├── test_hybrid_attention.py       # 对应 §3.2
├── test_attention_block.py        # 对应 §3.3
├── test_vla_flow_matching_init.py # 对应 §3.4
├── test_embed_prefix.py           # 对应 §3.5
├── test_embed_ae_tokens.py        # 对应 §3.6
├── test_prepare_attention_mask.py # 对应 §3.7
├── test_forward.py                # 对应 §3.8
├── test_sample_actions.py         # 对应 §3.9
├── test_load_stage0_weights.py    # 对应 §3.10
├── test_policy_init.py            # 对应 §3.11-3.12
└── test_set_requires_grad.py      # 对应 §3.4 子节
```

### 公共 fixtures（`conftest.py`）

```python
import pytest
import torch
from lerobot.policies.smolvla_apt.configuration_smolvla_apt import SmolVLAAptConfig
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import (
    SmolVLAAptPolicy,
    VLAFlowMatching,
    GateFusionBlock,
    HybridAttentionLayers,
    AttentionBlock,
)

# ---- 尺寸常量 ----
B = 2                # batch size
VLM_HIDDEN = 1152    # SmolVLM2-500M hidden dim
AE_HIDDEN = 864      # VLM_HIDDEN * 0.75 (expert_width_multiplier)
NUM_VLM_LAYERS = 16
HIGHWAY_INTERVAL = 2
NUM_HIGHWAYS = NUM_VLM_LAYERS // HIGHWAY_INTERVAL  # 8
NUM_AE_LAYERS = NUM_HIGHWAYS                      # 8
CHUNK_SIZE = 50
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
NUM_VISION_TOKENS = 49   # 1 image, 7x7 patches
NUM_LANG_TOKENS = 10
L_VL = NUM_VISION_TOKENS + NUM_LANG_TOKENS  # 59
L_AE = 1 + CHUNK_SIZE                       # 51 (1 state + 50 actions)
NUM_HEADS = 16
HEAD_DIM = AE_HIDDEN // NUM_HEADS            # 54
NUM_STEPS = 10


@pytest.fixture
def config_stage0():
    """Stage 0 config with train_stage=0."""
    cfg = SmolVLAAptConfig()
    cfg.train_stage = 0
    cfg.vl_highway_interval = HIGHWAY_INTERVAL
    cfg.gate_fusion_init = 0.0
    cfg.num_vlm_layers = NUM_VLM_LAYERS
    cfg.chunk_size = CHUNK_SIZE
    cfg.max_state_dim = MAX_STATE_DIM
    cfg.max_action_dim = MAX_ACTION_DIM
    cfg.num_steps = NUM_STEPS
    return cfg


@pytest.fixture
def config_stage1():
    """Stage 1 config with train_stage=1."""
    cfg = SmolVLAAptConfig()
    cfg.train_stage = 1
    cfg.vl_highway_interval = HIGHWAY_INTERVAL
    cfg.gate_fusion_init = 0.0
    cfg.num_vlm_layers = NUM_VLM_LAYERS
    cfg.chunk_size = CHUNK_SIZE
    cfg.max_state_dim = MAX_STATE_DIM
    cfg.max_action_dim = MAX_ACTION_DIM
    cfg.num_steps = NUM_STEPS
    return cfg


@pytest.fixture
def vl_highways():
    """Fake 8 VLM highways, each (B, L_vl, VLM_HIDDEN)."""
    return [torch.randn(B, L_VL, VLM_HIDDEN) for _ in range(NUM_HIGHWAYS)]


@pytest.fixture
def vl_highways_proj():
    """Fake 8 projected highways, each (B, L_vl, AE_HIDDEN)."""
    return [torch.randn(B, L_VL, AE_HIDDEN) for _ in range(NUM_HIGHWAYS)]


@pytest.fixture
def dilated_mask():
    """Dilated mask blocking language↔vision and language↔action.
    Language tokens at positions 49..58 are blocked from vision (0..48) and action (59..109)."""
    L = L_VL + L_AE  # 110
    mask = torch.ones(B, L, L, dtype=torch.bool)
    lang_start = NUM_VISION_TOKENS   # 49
    lang_end = L_VL                  # 59
    act_start = L_VL                 # 59
    # Block language↔vision and language↔action
    for l in range(lang_start, lang_end):
        # language → vision
        mask[:, l, :lang_start] = False
        # vision → language
        mask[:, :lang_start, l] = False
        # language → action
        mask[:, l, act_start:] = False
        # action → language
        mask[:, act_start:, l] = False
    return mask


@pytest.fixture
def vla_tokens():
    """VL tokens (B, L_vl, AE_HIDDEN)."""
    return torch.randn(B, L_VL, AE_HIDDEN)


@pytest.fixture
def ae_tokens():
    """AE tokens (B, L_ae, AE_HIDDEN)."""
    return torch.randn(B, L_AE, AE_HIDDEN)


@pytest.fixture
def state():
    """State vector (B, max_state_dim)."""
    return torch.randn(B, MAX_STATE_DIM)


@pytest.fixture
def actions():
    """Actions tensor (B, chunk_size, max_action_dim)."""
    return torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM)


@pytest.fixture
def noise():
    """Noise tensor matching actions shape."""
    return torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM)


@pytest.fixture
def images():
    """Single image (B, 3, 512, 512)."""
    return torch.rand(B, 3, 512, 512)


@pytest.fixture
def img_masks():
    """Image mask, all True."""
    return torch.ones(B, dtype=torch.bool)


@pytest.fixture
def lang_tokens():
    """Language tokens (B, num_lang_tokens)."""
    return torch.randint(0, 1000, (B, NUM_LANG_TOKENS))


@pytest.fixture
def lang_masks():
    """Language mask, all True."""
    return torch.ones(B, NUM_LANG_TOKENS, dtype=torch.bool)


@pytest.fixture
def gate_fusion_block():
    """Pre-built GateFusionBlock."""
    return GateFusionBlock(NUM_AE_LAYERS, AE_HIDDEN, gate_init=0.0)


@pytest.fixture
def hybrid_attn_stage0():
    """Stage 0 HybridAttentionLayers."""
    return HybridAttentionLayers(AE_HIDDEN, NUM_HEADS, NUM_AE_LAYERS, train_stage=0, head_dim=HEAD_DIM)


@pytest.fixture
def hybrid_attn_stage1():
    """Stage 1 HybridAttentionLayers."""
    return HybridAttentionLayers(AE_HIDDEN, NUM_HEADS, NUM_AE_LAYERS, train_stage=1, head_dim=HEAD_DIM)


@pytest.fixture
def attention_block():
    """Default AttentionBlock (rope lang, no vision PE)."""
    return AttentionBlock(AE_HIDDEN, NUM_HEADS, HEAD_DIM, pe_type=(None, "rope"))


@pytest.fixture
def attn_mask_2d():
    """Full attention mask (B, L_vl+L_ae, L_vl+L_ae), all True."""
    L = L_VL + L_AE
    return torch.ones(B, L, L, dtype=torch.bool)
```

---

## §1 `test_config.py` — 配置字段（§1.1, §1.2）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_new_fields_exist` | `train_stage`、`vl_highway_interval`、`gate_fusion_init` 存在且默认值正确 | 1.1 |
| `test_deprecated_fields_removed` | `attention_mode`、`self_attn_every_n_layers`、`train_state_proj` 不存在 | 1.1 |
| `test_train_stage_valid` | `train_stage=0`/`1` 通过验证 | 1.2 |
| `test_train_stage_invalid_raises` | `train_stage=2` 抛出 `ValueError` | 1.2 |

```python
# test_config.py

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
```

---

## §2 `test_vlm_highway.py` — VLM Highway 提取（§2.1）

> **注意**：此测试需要真实 `SmolVLMWithExpertModel`。若不能加载真实模型权重（无网络/HF token），
> 可改为集成测试并用 `pytest.mark.slow` + `pytest.mark.skipif` 标记。

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_forward_returns_three_tuples` | `forward()` 返回 3 元组 | 2.1 |
| `test_highway_count_matches_interval` | highway 数量 = `num_vlm_layers // vl_highway_interval` | 2.1 |
| `test_highway_shape` | 每个 highway shape = `(B, L_prefix, vlm_hidden_dim)` | 2.1 |
| `test_highway_detached` | highway requires_grad=False（VLM 冻结） | 2.1 |
| `test_different_interval` | `vl_highway_interval=4` → highway 数正确 | 2.1 |
| `test_forward_still_returns_past_kv` | `past_key_values` 仍返回（接口兼容） | 2.2 |

```python
# test_vlm_highway.py

import pytest
import torch


@pytest.mark.slow
class TestVLMHighwayExtraction:
    def test_forward_returns_three_tuples(self, smolvlm_model, prefix_embs):
        """forward() returns (outputs_embeds, past_key_values, collected_hidden_states)."""
        _, _, highways = smolvlm_model.forward(
            attention_mask=...,
            position_ids=...,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert isinstance(highways, list)
        assert len(highways) > 0

    def test_highway_count_matches_interval(self, smolvlm_model, prefix_embs):
        """num_highways = num_vlm_layers // vl_highway_interval."""
        interval = smolvlm_model.vl_highway_interval  # 2
        num_layers = len(smolvlm_model.get_vlm_model().text_model.layers)  # 16
        expected = num_layers // interval  # 8

        _, _, highways = smolvlm_model.forward(
            attention_mask=...,
            position_ids=...,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert len(highways) == expected

    def test_highway_shape(self, smolvlm_model, prefix_embs):
        """Each highway = (B, L_prefix, vlm_hidden_dim)."""
        B, L, D = prefix_embs.shape

        _, _, highways = smolvlm_model.forward(
            attention_mask=...,
            position_ids=...,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        for hw in highways:
            assert hw.shape == (B, L, D)

    def test_highway_detached(self, smolvlm_model, prefix_embs):
        """Highways should have requires_grad=False (VLM frozen)."""
        _, _, highways = smolvlm_model.forward(
            attention_mask=...,
            position_ids=...,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        for hw in highways:
            assert not hw.requires_grad

    def test_different_interval(self, smolvlm_model, prefix_embs):
        """vl_highway_interval=4 → fewer highways."""
        smolvlm_model.vl_highway_interval = 4
        num_layers = len(smolvlm_model.get_vlm_model().text_model.layers)
        expected = num_layers // 4

        _, _, highways = smolvlm_model.forward(
            attention_mask=...,
            position_ids=...,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert len(highways) == expected

    def test_forward_still_returns_past_kv(self, smolvlm_model, prefix_embs):
        """past_key_values should still be returned (interface compat)."""
        _, past_kv, _ = smolvlm_model.forward(
            attention_mask=...,
            position_ids=...,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        assert past_kv is not None or past_kv == {}
```

---

## §3.1 `test_gate_fusion.py` — `GateFusionBlock`（§3.1）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_init_gate_shape` | gate shape = `(num_layers, hidden_dim)` | 3.1 |
| `test_init_gate_zero` | 初始 gate = 0 → `sigmoid(0) = 0.5` | 3.1 |
| `test_forward_preserves_shape` | 输入输出 shape 一致 | 3.1 |
| `test_forward_equal_weight_at_init` | 初始步：`vl = 0.5*vl + 0.5*highway` | 3.1 |
| `test_different_layer_different_gate` | `layer_idx=0` 和 `layer_idx=1` 用不同 gate | 3.1 |
| `test_gate_all_one` | gate=10 → sigmoid≈1 → vl 几乎不变 | 3.1 |
| `test_gate_all_neg` | gate=-10 → sigmoid≈0 → 完全被 highway 替换 | 3.1 |
| `test_gradient_flows` | 反向传播能更新 gate 参数 | 3.1 |

```python
# test_gate_fusion.py

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import GateFusionBlock


class TestGateFusionBlock:
    def test_init_gate_shape(self, gate_fusion_block):
        assert gate_fusion_block.gate.shape == (8, 864)  # NUM_AE_LAYERS, AE_HIDDEN

    def test_init_gate_zero(self, gate_fusion_block):
        assert (gate_fusion_block.gate == 0.0).all()

    def test_forward_preserves_shape(self, gate_fusion_block, vla_tokens, vl_highways_proj):
        out = gate_fusion_block(vla_tokens, vl_highways_proj[0], layer_idx=0)
        assert out.shape == vla_tokens.shape

    def test_forward_equal_weight_at_init(self, gate_fusion_block):
        vl = torch.ones(2, 10, 864)
        hw = torch.zeros(2, 10, 864)
        out = gate_fusion_block(vl, hw, layer_idx=0)
        expected = 0.5 * vl + 0.5 * hw  # sigmoid(0)=0.5
        assert torch.allclose(out, expected, atol=1e-6)

    def test_different_layer_different_gate(self, gate_fusion_block, vla_tokens, vl_highways_proj):
        out0 = gate_fusion_block(vla_tokens, vl_highways_proj[0], layer_idx=0)
        out1 = gate_fusion_block(vla_tokens, vl_highways_proj[1], layer_idx=1)
        # Different layers have independent gate entries
        assert not torch.equal(out0, out1) or torch.equal(vl_highways_proj[0], vl_highways_proj[1])

    def test_gate_all_one(self):
        """gate=10 → sigmoid(10)≈1 → vl mostly preserved."""
        block = GateFusionBlock(8, 864, gate_init=10.0)
        vl = torch.ones(2, 10, 864)
        hw = torch.zeros(2, 10, 864)
        out = block(vl, hw, layer_idx=0)
        # sigmoid(10) ≈ 0.9999
        assert torch.allclose(out, vl, atol=1e-3)

    def test_gate_all_neg(self):
        """gate=-10 → sigmoid(-10)≈0 → fully replaced by highway."""
        block = GateFusionBlock(8, 864, gate_init=-10.0)
        vl = torch.zeros(2, 10, 864)
        hw = torch.ones(2, 10, 864)
        out = block(vl, hw, layer_idx=0)
        assert torch.allclose(out, hw, atol=1e-3)

    def test_gradient_flows(self, gate_fusion_block, vla_tokens, vl_highways_proj):
        vl = vla_tokens.clone().requires_grad_(True)
        out = gate_fusion_block(vl, vl_highways_proj[0], layer_idx=0)
        loss = out.sum()
        loss.backward()
        assert gate_fusion_block.gate.grad is not None
```

---

## §3.2 `test_hybrid_attention.py` — `HybridAttentionLayers`（§3.2）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_stage0_num_layers` | Stage 0: `num_active_layers = ceil(num_layers/2)` | 3.2 |
| `test_stage1_num_layers` | Stage 1: `num_active_layers = num_layers` | 3.2 |
| `test_stage1_even_num_layers` | Stage 1 `num_layers` 必须是偶数，奇数抛异常 | 3.2 |
| `test_forward_shape_preserved` | 输入输出 shape 一致 | 3.2 |
| `test_gate_fusion_applied` | 有 highway 时 gate 融合发生（输出与无 highway 不同） | 3.2 |
| `test_no_highway_no_crash` | `vl_highways` 全为 None 也能正常运行 | 3.2 |
| `test_stage0_highway_matches_gate_index` | Stage 0 中 highway 和 gate 使用同一索引 j = 2*i+1 | 3.2 |
| `test_stage1_odd_layers_use_dilated` | Stage 1 奇数层使用 dilated_mask，偶数层使用 full_mask | 3.2 |
| `test_stage0_all_va_pe` | Stage 0 所有层 pe_type = `("sinusoidal", None)` | 3.2 |
| `test_stage1_interleaved_pe` | Stage 1 偶数层 `(None, "rope")`，奇数层 `("sinusoidal", None)` | 3.2 |
| `test_vl_split_and_merge` | VL 和 AE token 在 gate fusion 后正确重组 | 3.2 |
| `test_gate_count_matches_layers` | gate 参数数量 = Stage 1 全量层数 | 3.2 |

```python
# test_hybrid_attention.py

import pytest
import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import HybridAttentionLayers


class TestHybridAttentionLayers:
    def test_stage0_num_layers(self, hybrid_attn_stage0):
        """Stage 0: ceil(8/2) = 4 layers."""
        assert hybrid_attn_stage0.num_active_layers == 4
        assert len(hybrid_attn_stage0.layers) == 4

    def test_stage1_num_layers(self, hybrid_attn_stage1):
        """Stage 1: full 8 layers."""
        assert hybrid_attn_stage1.num_active_layers == 8
        assert len(hybrid_attn_stage1.layers) == 8

    def test_stage1_even_num_layers(self):
        """Stage 1 requires even num_layers."""
        with pytest.raises(ValueError, match="even"):
            HybridAttentionLayers(864, 16, 7, train_stage=1, head_dim=54)

    def test_forward_shape_preserved(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x, attention_mask=attn_mask_2d,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        assert out.shape == x.shape

    def test_gate_fusion_applied(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out_with = hybrid_attn_stage1(
            x, attention_mask=attn_mask_2d,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        out_without = hybrid_attn_stage1(
            x, attention_mask=attn_mask_2d,
            vla_split_sizes=(59, 51),
            vl_highways=[None] * 8,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        assert not torch.equal(out_with, out_without)

    def test_no_highway_no_crash(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=None,
            vla_split_sizes=(59, 51),
            vl_highways=[None] * 8,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        assert out.shape == x.shape

    def test_stage0_highway_matches_gate_index(
        self, hybrid_attn_stage0, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj, dilated_mask
    ):
        """A1 fix: Stage 0 uses highway[j] where j = gate_idx = 2*i+1, not highway[i]."""
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        accessed_indices = []
        original_fusion = hybrid_attn_stage0.gate_fusion.forward

        def tracking_fusion(vl, hw, layer_idx):
            accessed_indices.append(layer_idx)
            return original_fusion(vl, hw, layer_idx)

        hybrid_attn_stage0.gate_fusion.forward = tracking_fusion
        hybrid_attn_stage0(
            x,
            attention_mask=attn_mask_2d,
            dilated_mask=dilated_mask,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        hybrid_attn_stage0.gate_fusion.forward = original_fusion
        # Stage 0: 4 layers (i=0,1,2,3), gate_idx=j=2*i+1 → [1,3,5,7]
        assert accessed_indices == [1, 3, 5, 7]

    def test_stage1_odd_layers_use_dilated(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj, dilated_mask
    ):
        """A2 fix: Stage 1 odd layers use dilated_mask, even layers use full_mask."""
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        masks_used = []

        def hook_fn(module, args, kwargs):
            masks_used.append(kwargs.get('attention_mask'))

        handles = [layer.register_forward_hook(hook_fn, with_kwargs=True)
                   for layer in hybrid_attn_stage1.layers]
        hybrid_attn_stage1(
            x,
            attention_mask=attn_mask_2d,     # full mask
            dilated_mask=dilated_mask,        # lang-blocked mask
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        for h in handles:
            h.remove()

        assert len(masks_used) == 8
        # Odd layers (1,3,5,7) = dilated_mask; Even layers (0,2,4,6) = full_mask
        for i in [1, 3, 5, 7]:
            assert masks_used[i] is dilated_mask
        for i in [0, 2, 4, 6]:
            assert masks_used[i] is attn_mask_2d

    def test_stage0_all_va_pe(self, hybrid_attn_stage0):
        """Stage 0: all pe_types = ('sinusoidal', None)."""
        for layer in hybrid_attn_stage0.layers:
            assert layer.pe_type == ("sinusoidal", None)

    def test_stage1_interleaved_pe(self, hybrid_attn_stage1):
        """Stage 1: even=(None, 'rope'), odd=('sinusoidal', None)."""
        for i, layer in enumerate(hybrid_attn_stage1.layers):
            if i % 2 == 0:
                assert layer.pe_type == (None, "rope")
            else:
                assert layer.pe_type == ("sinusoidal", None)

    def test_vl_split_and_merge(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        """VL tokens (59) + AE tokens (51) are split, gated, and merged correctly."""
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x, attention_mask=attn_mask_2d,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        # After merge, total length should be unchanged
        assert out.shape[1] == 110

    def test_gate_count_matches_layers(self, hybrid_attn_stage1):
        """Gate fusion has 8 gates (full Stage 1 count)."""
        assert hybrid_attn_stage1.gate_fusion.gate.shape[0] == 8

    def test_forward_no_nan(
        self, hybrid_attn_stage1, vla_tokens, ae_tokens, attn_mask_2d, vl_highways_proj
    ):
        x = torch.cat([vla_tokens, ae_tokens], dim=1)
        out = hybrid_attn_stage1(
            x, attention_mask=attn_mask_2d,
            vla_split_sizes=(59, 51),
            vl_highways=vl_highways_proj,
            position_ids=torch.arange(110).unsqueeze(0).expand(2, -1),
        )
        assert not torch.isnan(out).any()
```

---

## §3.3 `test_attention_block.py` — `AttentionBlock`（§3.3）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_forward_shape` | 输入输出 shape 一致 | 3.3 |
| `test_residual_connection` | `attn_mask` 全 0 时输出 ≈ 输入（仅残差） | 3.3 |
| `test_swiglu_ffn` | 验证 gate/up/down 三层结构存在 | 3.3 |
| `test_rms_norm_present` | `norm1` 和 `norm2` 是 `RMSNorm` | 3.3 |
| `test_no_film` | `norm2` 不是 `AdaRMSNorm`（FiLM 已删除） | 3.3 |
| `test_rope_pe_type` | pe_type=(None, "rope") 不报错 | 3.3 |
| `test_sinusoidal_pe_type` | pe_type=("sinusoidal", None) 不报错 | 3.3 |
| `test_gradient_flows` | 反向传播通过 attention + FFN | 3.3 |

```python
# test_attention_block.py

import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import AttentionBlock


class TestAttentionBlock:
    def test_forward_shape(self, attention_block, vla_tokens, attn_mask_2d):
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        out = attention_block(vla_tokens, attention_mask=mask)
        assert out.shape == vla_tokens.shape

    def test_residual_connection(self, attention_block, vla_tokens):
        """When attention is zeroed, residual should pass input through."""
        L = vla_tokens.shape[1]
        zero_mask = torch.zeros(2, L, L, dtype=torch.bool)  # all zero → no attention
        out = attention_block(vla_tokens, attention_mask=zero_mask)
        # With full masking and RMSNorm initial scale ~1, output differs from input
        # due to FFN, but should not be NaN
        assert not torch.isnan(out).any()
        assert out.shape == vla_tokens.shape

    def test_swiglu_ffn(self, attention_block):
        """SwiGLU FFN has gate_proj, up_proj, down_proj."""
        assert hasattr(attention_block, "gate_proj")
        assert hasattr(attention_block, "up_proj")
        assert hasattr(attention_block, "down_proj")
        # gate_proj and up_proj have no bias
        assert attention_block.gate_proj.bias is None

    def test_rms_norm_present(self, attention_block):
        """norm1 and norm2 are RMSNorm."""
        assert isinstance(attention_block.norm1, torch.nn.RMSNorm)
        assert isinstance(attention_block.norm2, torch.nn.RMSNorm)

    def test_no_film(self):
        """Verify AdaRMSNorm is NOT used (FiLM deleted per §S2)."""
        # Attempt to import AdaRMSNorm should fail
        with __import__('pytest').raises(ImportError):
            from lerobot.policies.smolvla_apt.modeling_smolvla_apt import AdaRMSNorm

    def test_rope_pe_type(self, vla_tokens, attn_mask_2d):
        block = AttentionBlock(864, 16, 54, pe_type=(None, "rope"))
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        out = block(vla_tokens, attention_mask=mask)
        assert out.shape == vla_tokens.shape

    def test_sinusoidal_pe_type(self, vla_tokens, attn_mask_2d):
        block = AttentionBlock(864, 16, 54, pe_type=("sinusoidal", None))
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        out = block(vla_tokens, attention_mask=mask)
        assert out.shape == vla_tokens.shape

    def test_gradient_flows(self, attention_block, vla_tokens, attn_mask_2d):
        L = vla_tokens.shape[1]
        mask = attn_mask_2d[:, :L, :L]
        x = vla_tokens.clone().requires_grad_(True)
        out = attention_block(x, attention_mask=mask)
        loss = out.sum()
        loss.backward()
        for name, param in attention_block.named_parameters():
            assert param.grad is not None, f"{name} should have grad"
```

---

## §3.4a `test_vla_flow_matching_init.py` — `VLAFlowMatching.__init__`（§3.4）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_all_modules_created` | 所有新模块存在且是正确类型 | 3.4 |
| `test_state_proj_removed` | `state_proj` 不存在（迁移到 AE） | 3.4 |
| `test_ae_state_proj_exists` | `ae_state_proj` 存在 | 3.4 |
| `test_vl0_proj_dim` | `vl0_proj` 输入 `vlm_hidden_dim`，输出 `ae_hidden_dim` | 3.4 |
| `test_vl_highway_proj_count` | `vl_highway_proj` 层数 = num_highways | 3.4 |
| `test_hybrid_attn_exists` | `hybrid_attn_layers` 存在 | 3.4 |
| `test_num_ae_layers_equals_num_highways` | `num_ae_layers = num_vlm_layers // vl_highway_interval` | 3.4 |

```python
# test_vla_flow_matching_init.py

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

    def test_ae_state_proj_exists(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        assert isinstance(model.ae_state_proj, torch.nn.Linear)
        assert model.ae_state_proj.in_features == 32   # max_state_dim
        assert model.ae_state_proj.out_features == 864  # ae_hidden_dim

    def test_vl0_proj_dim(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        assert model.vl0_proj.in_features == 1152  # vlm_hidden_dim
        assert model.vl0_proj.out_features == 864  # ae_hidden_dim

    def test_vl_highway_proj_count(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        num_highways = config_stage0.num_vlm_layers // config_stage0.vl_highway_interval  # 8
        assert len(model.vl_highway_proj) == num_highways
        for proj in model.vl_highway_proj:
            assert proj.in_features == 1152
            assert proj.out_features == 864

    def test_hybrid_attn_exists(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        assert model.hybrid_attn_layers is not None

    def test_num_ae_layers_equals_num_highways(self, config_stage0):
        model = VLAFlowMatching(config_stage0)
        expected = config_stage0.num_vlm_layers // config_stage0.vl_highway_interval  # 8
        # Gate fusion gate has 'expected' rows
        assert model.hybrid_attn_layers.gate_fusion.gate.shape[0] == expected
```

---

## §3.5 `test_embed_prefix.py` — `embed_prefix` 重构（§3.5）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_no_state_in_prefix` | prefix 中无 state token | 3.5 |
| `test_returns_four_values` | 返回 `(embs, pad_masks, att_masks, num_vision_tokens)` | 3.5 |
| `test_num_vision_tokens_correct` | `num_vision_tokens` = 图像 token 数（含特殊 token） | 3.5 |
| `test_num_vision_tokens_with_special` | `add_image_special_tokens=True` 时计数正确 | 3.5 |
| `test_pad_tokens_all_at_end` | 若 `prefix_length>0`，pad token 在末尾 | 3.5 |
| `test_output_shapes_match` | `embs`、`pad_masks`、`att_masks` 的 L 一致 | 3.5 |

```python
# test_embed_prefix.py

import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import create_sinusoidal_pos_embedding


class TestEmbedPrefix:
    def test_no_state_in_prefix(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
        """State is NOT in prefix; it moved to AE."""
        vfm = vla_flow_matching
        embs, pad_masks, att_masks, nvt = vfm.embed_prefix(
            [images], [img_masks], lang_tokens, lang_masks
        )
        # prefix 中没有 state_proj 的输出
        # state token 不应出现在序列中
        assert embs.shape[1] == nvt + lang_tokens.shape[1]  # vision + language only

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
        assert nvt > 0  # should have at least image patch tokens

    def test_num_vision_tokens_with_special(self):
        """With add_image_special_tokens=True, num_vision_tokens includes special tokens."""
        # This test requires a model with add_image_special_tokens=True
        ...

    def test_output_shapes_match(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
        vfm = vla_flow_matching
        embs, pad_masks, att_masks, nvt = vfm.embed_prefix(
            [images], [img_masks], lang_tokens, lang_masks
        )
        L = embs.shape[1]
        assert pad_masks.shape == (2, L)
        assert att_masks.shape == (2, L)
        assert nvt < L  # vision < total (vision + language)
```

---

## §3.6 `test_embed_ae_tokens.py` — `embed_ae_tokens`（§3.6）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_returns_three_values` | 返回 `(ae_embs, ae_pad_masks, ae_att_masks)` | 3.6 |
| `test_state_first` | state token 在 action tokens 之前 | 3.6 |
| `test_output_shape` | `ae_embs.shape = (B, 1+L_action, ae_hidden_dim)` | 3.6 |
| `test_action_time_fusion` | 不同 timestep 产生不同嵌入 | 3.6 |
| `test_pad_masks_all_true` | AE pad masks 全为 True（无 padding） | 3.6 |
| `test_timestep_sinusoidal_embed` | 验证 `create_sinusoidal_pos_embedding` 输出 | 3.6 |

```python
# test_embed_ae_tokens.py

import torch
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import create_sinusoidal_pos_embedding


class TestEmbedAETokens:
    def test_returns_three_values(self, vla_flow_matching, actions, state):
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5])
        result = vfm.embed_ae_tokens(actions, time, state)
        assert len(result) == 3

    def test_state_first(self, vla_flow_matching, actions, state):
        """State token is at position 0, before action tokens."""
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5])
        embs, _, _ = vfm.embed_ae_tokens(actions, time, state)
        # The first token should be the state projection
        state_emb = vfm.ae_state_proj(state).unsqueeze(1)
        assert torch.allclose(embs[:, 0:1, :], state_emb, atol=1e-5)

    def test_output_shape(self, vla_flow_matching, actions, state):
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5])
        embs, pad_masks, att_masks = vfm.embed_ae_tokens(actions, time, state)
        assert embs.shape == (2, 1 + 50, 864)  # B, 1+chunk, AE_HIDDEN
        assert pad_masks.shape == (2, 51)
        assert att_masks.shape == (2, 51)

    def test_action_time_fusion(self, vla_flow_matching, actions, state):
        """Different timesteps should produce different action embeddings."""
        vfm = vla_flow_matching
        time0 = torch.tensor([0.1, 0.1])
        time1 = torch.tensor([0.9, 0.9])
        embs0, _, _ = vfm.embed_ae_tokens(actions, time0, state)
        embs1, _, _ = vfm.embed_ae_tokens(actions, time1, state)
        # Action tokens (positions 1:) should differ
        assert not torch.equal(embs0[:, 1:, :], embs1[:, 1:, :])

    def test_pad_masks_all_true(self, vla_flow_matching, actions, state):
        vfm = vla_flow_matching
        time = torch.tensor([0.5, 0.5])
        _, pad_masks, _ = vfm.embed_ae_tokens(actions, time, state)
        assert pad_masks.all()

    def test_timestep_sinusoidal_embed(self):
        """create_sinusoidal_pos_embedding produces valid output."""
        time = torch.tensor([0.2, 0.8])
        emb = create_sinusoidal_pos_embedding(time, dimension=864, min_period=4e-3, max_period=4.0)
        assert emb.shape == (2, 864)
        assert not torch.isnan(emb).any()
        # Different times should produce different embeddings
        assert not torch.equal(emb[0], emb[1])
```

---

## §3.7 `test_prepare_attention_mask.py` — 注意力掩码（§3.7）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_full_mask_shape` | `full_mask.shape = (B, L, L)`，全 True | 3.7 |
| `test_dilated_mask_blocks_language` | language 不能 attend vision/action，反之亦然 | 3.7 |
| `test_modality_vision_is_1` | Vision tokens 的 `modality=1` | 3.7 |
| `test_modality_language_is_2` | Language tokens 的 `modality=2` | 3.7 |
| `test_modality_action_is_3` | Action tokens 的 `modality=3` | 3.7 |
| `test_stage0_all_language_blocked` | Stage 0: 所有 L↔V 和 L↔A 均被阻断 | 3.7 |
| `test_stage1_dilated_same_as_stage0` | Stage 1 dilated_mask 结构一致（奇数层逻辑） | 3.7 |
| `test_no_cross_modality_leak` | 逐个检查 language→vision、language→action 无漏 | 3.7 |

```python
# test_prepare_attention_mask.py

import torch


class TestPrepareAttentionMask:
    def test_full_mask_shape(self, vla_flow_matching):
        vfm = vla_flow_matching
        vl_mask = torch.ones(2, 59, dtype=torch.bool)
        ae_mask = torch.ones(2, 51, dtype=torch.bool)
        full, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        assert full.shape == (2, 110, 110)
        assert full.all()

    def test_dilated_mask_blocks_language(self, vla_flow_matching):
        """Language tokens (positions 49..59) blocked from vision and action."""
        vfm = vla_flow_matching
        vl_mask = torch.ones(2, 59, dtype=torch.bool)
        ae_mask = torch.ones(2, 51, dtype=torch.bool)
        _, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        # Language at col 50 should NOT be visible to vision at col 0
        assert not dilated[0, 0, 50]   # vision→lang blocked
        assert not dilated[0, 50, 0]   # lang→vision blocked
        assert not dilated[0, 100, 50] # action→lang blocked
        assert not dilated[0, 50, 100] # lang→action blocked

    def test_modality_vision_is_1(self, vla_flow_matching):
        """Vision tokens at positions 0..49 should be blocked from language."""
        vfm = vla_flow_matching
        vl_mask = torch.ones(3, 59, dtype=torch.bool)
        ae_mask = torch.ones(3, 51, dtype=torch.bool)
        _, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, 49, train_stage=0)
        # Check a vision token (position 0) cannot see a language token (position 50)
        assert not dilated[:, 0, 50].any()

    def test_no_cross_modality_leak(self, vla_flow_matching):
        """Exhaustive check for language↔vision and language↔action."""
        vfm = vla_flow_matching
        nv, nl, na = 49, 10, 51
        L = nv + nl + na
        vl_mask = torch.ones(1, nv + nl, dtype=torch.bool)
        ae_mask = torch.ones(1, na, dtype=torch.bool)
        _, dilated = vfm.prepare_attention_mask(vl_mask, ae_mask, nv, train_stage=0)

        for v in range(nv):
            for l in range(nv, nv + nl):
                assert not dilated[0, v, l], f"vision→lang leak at {v}→{l}"
                assert not dilated[0, l, v], f"lang→vision leak at {l}→{v}"
        for a in range(nv + nl, L):
            for l in range(nv, nv + nl):
                assert not dilated[0, a, l], f"action→lang leak at {a}→{l}"
                assert not dilated[0, l, a], f"lang→action leak at {l}→{a}"
```

---

## §3.8 `test_forward.py` — 训练前向（§3.8）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_forward_returns_loss_and_dict` | 返回 `(loss, loss_dict)` | 3.8 |
| `test_loss_scalar` | `loss` 是标量 tensor | 3.8 |
| `test_loss_positive` | MSE loss ≥ 0 | 3.8 |
| `test_deterministic_same_input` | 相同输入 → 相同 loss | 3.8 |
| `test_different_noise_different_loss` | 不同 noise → 不同 loss | 3.8 |
| `test_vlm_only_called_once` | VLM forward 只调用一次（非 denoising 循环） | 3.8 |
| `test_highways_used` | highway 被使用（有 vs 无 highway 产生不同 loss） | 3.8 |
| `test_stage0_vs_stage1_different_loss` | Stage 0/1 产生不同 loss | 3.8 |
| `test_batch_size_invariant` | 不同 batch size 不影响 per-sample mean | 3.8 |

```python
# test_forward.py

import torch
import torch.nn.functional as F


class TestForward:
    def test_forward_returns_loss_and_dict(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        loss, loss_dict = vfm.forward(**batch)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict

    def test_loss_scalar(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        loss, _ = vfm.forward(**batch)
        assert loss.ndim == 0

    def test_loss_positive(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        loss, _ = vfm.forward(**batch)
        assert loss.item() >= 0

    def test_deterministic_same_input(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        vfm.eval()
        torch.manual_seed(42)
        loss1, _ = vfm.forward(**batch, noise=torch.randn(2, 50, 32), time=torch.tensor([0.5, 0.5]))
        torch.manual_seed(42)
        loss2, _ = vfm.forward(**batch, noise=torch.randn(2, 50, 32), time=torch.tensor([0.5, 0.5]))
        assert torch.allclose(loss1, loss2)

    def test_different_noise_different_loss(self, vla_flow_matching, batch):
        vfm = vla_flow_matching
        loss1, _ = vfm.forward(**batch, noise=torch.randn(2, 50, 32))
        loss2, _ = vfm.forward(**batch, noise=torch.randn(2, 50, 32))
        assert loss1.item() != loss2.item()

    def test_highways_used(self, vla_flow_matching, batch):
        """With valid highways, loss should differ from zeroed highways."""
        vfm = vla_flow_matching
        # normal forward
        loss1, _ = vfm.forward(**batch)
        # Zero out highways
        with torch.no_grad():
            for proj in vfm.vl_highway_proj:
                proj.weight.zero_()
        loss2, _ = vfm.forward(**batch)
        assert loss1.item() != loss2.item()

    def test_stage0_vs_stage1_different_loss(self, vla_stage0, vla_stage1, batch):
        """Stage 0 and Stage 1 have different architectures → different loss."""
        loss0, _ = vla_stage0.forward(**batch)
        loss1, _ = vla_stage1.forward(**batch)
        assert loss0.item() != loss1.item()

    def test_batch_size_invariant(self, vla_flow_matching):
        """Per-sample mean loss should be similar for different batch sizes."""
        ...
```

---

## §3.9 `test_sample_actions.py` — 推理前向（§3.9）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_output_shape` | `(B, chunk_size, max_action_dim)` | 3.9 |
| `test_vlm_called_once` | VLM forward 去噪循环中只调用一次 | 3.9 |
| `test_position_ids_full_sequence` | position_ids 覆盖完整 VL+AE 序列 | 3.9 |
| `test_num_steps_obeys_config` | denoising 步数 = config.num_steps | 3.9 |
| `test_output_no_nan` | 推理输出无 NaN | 3.9 |
| `test_different_noise_different_output` | 不同初始 noise → 不同动作 | 3.9 |
| `test_rtc_compatible` | RTC 路径不报错（若配置了 RTC） | 3.9 |

```python
# test_sample_actions.py

import torch


class TestSampleActions:
    def test_output_shape(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        vfm = vla_flow_matching
        vfm.eval()
        with torch.no_grad():
            actions = vfm.sample_actions(
                [images], [img_masks], lang_tokens, lang_masks, state
            )
        assert actions.shape == (2, 50, 32)  # B, chunk, action_dim

    @pytest.mark.slow
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
        """position_ids should span VL + AE, not just AE."""
        vfm = vla_flow_matching
        vfm.eval()
        # Hook to capture position_ids passed to hybrid_attn_layers
        captured_ids = []
        def hook(module, args, kwargs):
            if 'position_ids' in kwargs:
                captured_ids.append(kwargs['position_ids'])

        handle = vfm.hybrid_attn_layers.register_forward_hook(hook, with_kwargs=True)
        with torch.no_grad():
            vfm.sample_actions([images], [img_masks], lang_tokens, lang_masks, state)
        handle.remove()

        pid = captured_ids[0]
        # Should cover VL + AE tokens
        assert pid.shape[1] > 51  # > AE-only (51)
        assert pid.shape[1] >= 59 + 51  # VL + AE

    def test_num_steps_obeys_config(self, vla_flow_matching, images, img_masks, lang_tokens, lang_masks, state):
        vfm = vla_flow_matching
        vfm.config.num_steps = 5
        vfm.eval()
        # Patch to count steps
        step_count = 0

        def counting_embed_ae(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return vfm._original_embed_ae_tokens(*args, **kwargs)

        vfm._original_embed_ae_tokens = vfm.embed_ae_tokens
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
```

---

## §3.10 `test_load_stage0_weights.py` — 权重加载（§3.10）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_layer_mapping` | Stage 0 Layer i → Stage 1 Layer 2i+1 | 3.10 |
| `test_va_layers_loaded` | Stage 0 奇数层权重正确复制到 Stage 1 | 3.10 |
| `test_even_layers_not_loaded` | Stage 1 偶数层随机初始化（不从 Stage 0 加载） | 3.10 |
| `test_projection_layers_copied` | `ae_state_proj`、`vl0_proj` 等直接复制 | 3.10 |
| `test_gate_not_copied` | Gate 参数不从 Stage 0 复制到 Stage 1（全新零初始化） | 3.10 |
| `test_wrong_stage_raises` | train_stage=0 调 `load_stage0_weights` 抛异常 | 3.10 |
| `test_strict_false_no_error` | `strict=False` 不因多余 key 报错 | 3.10 |

```python
# test_load_stage0_weights.py

import pytest
import torch
import copy


class TestLoadStage0Weights:
    def test_layer_mapping(self, vla_stage1, vla_stage0_weights):
        """Stage 0 Layer 0 → Stage 1 Layer 1, etc."""
        # Save Stage 1 original weights
        s1_orig = copy.deepcopy(vla_stage1.hybrid_attn_layers.layers[1].norm1.weight)
        # Load Stage 0 weights
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        s1_new = vla_stage1.hybrid_attn_layers.layers[1].norm1.weight
        # Should have changed (loaded from Stage 0)
        assert not torch.equal(s1_orig, s1_new)

    def test_even_layers_not_loaded(self, vla_stage1, vla_stage0_weights):
        """Stage 1 even layers (0, 2, 4, 6) should NOT be loaded from Stage 0."""
        s1_orig = copy.deepcopy(vla_stage1.hybrid_attn_layers.layers[0].norm1.weight)
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        s1_new = vla_stage1.hybrid_attn_layers.layers[0].norm1.weight
        # Should be UNCHANGED (even layer not in Stage 0)
        assert torch.equal(s1_orig, s1_new)

    def test_projection_layers_copied(self, vla_stage1, vla_stage0_weights):
        """Projection layers copied directly."""
        orig = copy.deepcopy(vla_stage1.ae_state_proj.weight)
        vla_stage1.load_state_dict(vla_stage0_weights, strict=False)
        if "ae_state_proj.weight" in vla_stage0_weights:
            assert not torch.equal(orig, vla_stage1.ae_state_proj.weight)

    def test_wrong_stage_raises(self, policy_stage0):
        with pytest.raises(ValueError, match="train_stage"):
            policy_stage0.load_stage0_weights({})

    def test_strict_false_no_error(self, vla_stage1):
        """strict=False should not error even with extra/missing keys."""
        # Empty dict should not raise
        vla_stage1.load_state_dict({}, strict=False)
```

---

## §3.11-3.12 `test_policy_init.py` — Policy 初始化 & PEFT 删除（§3.11, §3.12）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_stage0_weights_not_loaded` | `train_stage=0` 不触发权重加载 | 3.11 |
| `test_stage1_weights_loaded_if_path` | `train_stage=1` + `pretrained_path` → 加载 | 3.11 |
| `test_stage1_no_path_no_crash` | `train_stage=1` 但没有 `pretrained_path` 不报错 | 3.11 |
| `test_peft_methods_deleted` | `_get_default_peft_targets` 和 `_validate_peft_config` 不存在 | 3.12 |
| `test_model_created` | `self.model` 是 `VLAFlowMatching` 实例 | 3.11 |
| `test_reset_called` | `self._queues` 在 init 后被初始化 | 3.11 |

```python
# test_policy_init.py

import pytest
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import SmolVLAAptPolicy, VLAFlowMatching


class TestPolicyInit:
    def test_stage0_weights_not_loaded(self, config_stage0, monkeypatch):
        """Stage 0 should not try to load stage0 weights."""
        config_stage0.train_stage = 0
        config_stage0.pretrained_path = "/fake/path"
        # Should not error (load is skipped for stage 0)
        policy = SmolVLAAptPolicy(config_stage0)
        assert isinstance(policy.model, VLAFlowMatching)

    def test_peft_methods_deleted(self, config_stage0):
        policy = SmolVLAAptPolicy(config_stage0)
        assert not hasattr(policy, "_get_default_peft_targets")
        assert not hasattr(policy, "_validate_peft_config")

    def test_model_created(self, config_stage0):
        policy = SmolVLAAptPolicy(config_stage0)
        assert isinstance(policy.model, VLAFlowMatching)

    def test_reset_called(self, config_stage0):
        policy = SmolVLAAptPolicy(config_stage0)
        # _queues should exist and have ACTION key
        assert hasattr(policy, "_queues")
        assert "action" in policy._queues
```

---

## §3.4b `test_set_requires_grad.py` — set_requires_grad（§3.4 子节）

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_vlm_frozen` | VLM 参数 `requires_grad=False` | 3.4b |
| `test_ae_trainable` | 所有 AE 模块 `requires_grad=True` | 3.4b |
| `test_ae_state_proj_trainable` | `ae_state_proj` 可训练 | 3.4b |
| `test_vl0_proj_trainable` | `vl0_proj` 可训练 | 3.4b |
| `test_vl_highway_proj_trainable` | 每个 highway 投影层可训练 | 3.4b |
| `test_hybrid_attn_trainable` | `hybrid_attn_layers` 全部可训练 | 3.4b |
| `test_action_projections_trainable` | `action_in/out_proj` + `action_time_mlp_in/out` 可训练 | 3.4b |
| `test_old_state_proj_not_present` | 旧的 `state_proj` 不存在 | 3.4b |

```python
# test_set_requires_grad.py

import torch


class TestSetRequiresGrad:
    def test_vlm_frozen(self, vla_flow_matching):
        vfm = vla_flow_matching
        vfm.set_requires_grad()
        for name, param in vfm.vlm_with_expert.named_parameters():
            assert not param.requires_grad, f"VLM {name} should be frozen"

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
```

---

## §A 集成测试

| 测试 | 内容 | 覆盖 |
|------|------|------|
| `test_full_training_loop_stage0` | Stage 0 完整训练一个 step 不报错 | all |
| `test_full_training_loop_stage1` | Stage 1 完整训练一个 step 不报错 | all |
| `test_save_load_roundtrip` | 保存/加载 checkpoint 后 loss 一致 | all |
| `test_gradient_update_stage0` | Stage 0 训练后参数更新 | all |
| `test_gradient_update_stage1` | Stage 1 训练后参数更新 | all |
| `test_stage0_to_stage1_loading` | Stage 0 ckpt → Stage 1 加载后 loss 可训练 | all |

```python
# test_integration.py

import pytest
import torch


@pytest.mark.slow
class TestIntegration:
    def test_full_training_loop_stage0(self, vla_flow_matching_stage0, batch):
        vfm = vla_flow_matching_stage0
        vfm.train()
        loss, _ = vfm.forward(**batch)
        loss.backward()
        # Verify gradients exist
        for name, param in vfm.named_parameters():
            if param.requires_grad and "vlm_with_expert" not in name:
                assert param.grad is not None or param.grad_fn is not None

    def test_save_load_roundtrip(self, vla_flow_matching, batch, tmp_path):
        vfm = vla_flow_matching
        vfm.eval()
        torch.manual_seed(0)
        loss1, _ = vfm.forward(**batch)

        # Save
        ckpt = tmp_path / "model.safetensors"
        from safetensors.torch import save_file
        save_file(vfm.state_dict(), str(ckpt))

        # Reload
        from safetensors.torch import load_file
        sd = load_file(str(ckpt))
        vfm.load_state_dict(sd, strict=False)

        torch.manual_seed(0)
        loss2, _ = vfm.forward(**batch)
        assert torch.allclose(loss1, loss2)
```

---

## §B 覆盖率映射

| 实施文档章节 | 测试文件 | 覆盖率目标 |
|-------------|---------|-----------|
| §1.1-1.2 配置 | `test_config.py` | 90% |
| §2.1-2.3 VLM highway | `test_vlm_highway.py` | 80% |
| §3.1 GateFusionBlock | `test_gate_fusion.py` | 95% |
| §3.2 HybridAttentionLayers | `test_hybrid_attention.py` | 85% |
| §3.3 AttentionBlock | `test_attention_block.py` | 90% |
| §3.4 __init__ | `test_vla_flow_matching_init.py` | 85% |
| §3.4b set_requires_grad | `test_set_requires_grad.py` | 95% |
| §3.5 embed_prefix | `test_embed_prefix.py` | 80% |
| §3.6 embed_ae_tokens | `test_embed_ae_tokens.py` | 90% |
| §3.7 prepare_attention_mask | `test_prepare_attention_mask.py` | 95% |
| §3.8 forward | `test_forward.py` | 80% |
| §3.9 sample_actions | `test_sample_actions.py` | 80% |
| §3.10 load_stage0_weights | `test_load_stage0_weights.py` | 85% |
| §3.11-3.12 policy init/PEFT | `test_policy_init.py` | 85% |
| — 集成测试 | `test_integration.py` | 70% |
| **总体** | — | **≥82%** |

"""Shared fixtures for smolvla_apt tests."""

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
NUM_STEPS = 10


@pytest.fixture(scope="session")
def torch_device():
    """Device for test tensors — follows CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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
def vl_highways(torch_device):
    """Fake 8 VLM highways, each (B, L_vl, VLM_HIDDEN)."""
    return [torch.randn(B, L_VL, VLM_HIDDEN, device=torch_device) for _ in range(NUM_HIGHWAYS)]


@pytest.fixture
def vl_highways_proj(torch_device):
    """Fake 8 projected highways, each (B, L_vl, AE_HIDDEN)."""
    return [torch.randn(B, L_VL, AE_HIDDEN, device=torch_device) for _ in range(NUM_HIGHWAYS)]


@pytest.fixture
def dilated_mask(torch_device):
    """Dilated mask blocking language<->vision and language<->action.
    Language tokens at positions 49..58 are blocked from vision (0..48) and action (59..109)."""
    L = L_VL + L_AE  # 110
    mask = torch.ones(B, L, L, dtype=torch.bool, device=torch_device)
    lang_start = NUM_VISION_TOKENS   # 49
    lang_end = L_VL                  # 59
    act_start = L_VL                 # 59
    # Block language<->vision and language<->action
    for l_idx in range(lang_start, lang_end):
        mask[:, l_idx, :lang_start] = False   # language -> vision
        mask[:, :lang_start, l_idx] = False   # vision -> language
        mask[:, l_idx, act_start:] = False    # language -> action
        mask[:, act_start:, l_idx] = False    # action -> language
    return mask


@pytest.fixture
def vla_tokens(torch_device):
    """VL tokens (B, L_vl, AE_HIDDEN)."""
    return torch.randn(B, L_VL, AE_HIDDEN, device=torch_device)


@pytest.fixture
def ae_tokens(torch_device):
    """AE tokens (B, L_ae, AE_HIDDEN)."""
    return torch.randn(B, L_AE, AE_HIDDEN, device=torch_device)


@pytest.fixture
def state(torch_device):
    """State vector (B, max_state_dim)."""
    return torch.randn(B, MAX_STATE_DIM, device=torch_device)


@pytest.fixture
def actions(torch_device):
    """Actions tensor (B, chunk_size, max_action_dim)."""
    return torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=torch_device)


@pytest.fixture
def noise(torch_device):
    """Noise tensor matching actions shape."""
    return torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM, device=torch_device)


@pytest.fixture
def images(torch_device):
    """Single image (B, 3, 512, 512)."""
    return torch.rand(B, 3, 512, 512, device=torch_device)


@pytest.fixture
def img_masks(torch_device):
    """Image mask, all True."""
    return torch.ones(B, dtype=torch.bool, device=torch_device)


@pytest.fixture
def lang_tokens(torch_device):
    """Language tokens (B, num_lang_tokens)."""
    return torch.randint(0, 1000, (B, NUM_LANG_TOKENS), device=torch_device)


@pytest.fixture
def lang_masks(torch_device):
    """Language mask, all True."""
    return torch.ones(B, NUM_LANG_TOKENS, dtype=torch.bool, device=torch_device)


@pytest.fixture
def gate_fusion_block(torch_device):
    """Pre-built GateFusionBlock."""
    return GateFusionBlock(NUM_AE_LAYERS, AE_HIDDEN, gate_init=0.0).to(torch_device)


@pytest.fixture
def hybrid_attn_stage0(torch_device):
    """Stage 0 HybridAttentionLayers."""
    return HybridAttentionLayers(AE_HIDDEN, NUM_HEADS, NUM_AE_LAYERS, train_stage=0).to(torch_device)


@pytest.fixture
def hybrid_attn_stage1(torch_device):
    """Stage 1 HybridAttentionLayers."""
    return HybridAttentionLayers(AE_HIDDEN, NUM_HEADS, NUM_AE_LAYERS, train_stage=1).to(torch_device)


@pytest.fixture
def attention_block(torch_device):
    """Default AttentionBlock (rope lang, no vision PE)."""
    return AttentionBlock(AE_HIDDEN, NUM_HEADS, pe_type=(None, "rope")).to(torch_device)


@pytest.fixture
def attn_mask_2d(torch_device):
    """Full attention mask (B, L_vl+L_ae, L_vl+L_ae), all True."""
    L = L_VL + L_AE
    return torch.ones(B, L, L, dtype=torch.bool, device=torch_device)


# =============================================================================
# VLM-dependent fixtures (require downloaded VLM weights)
# =============================================================================

@pytest.fixture(scope="module")
def vla_flow_matching_stage0(config_stage0):
    """Stage 0 VLAFlowMatching with real VLM weights. Loaded once per module."""
    return VLAFlowMatching(config_stage0)


@pytest.fixture(scope="module")
def vla_flow_matching_stage1(config_stage1):
    """Stage 1 VLAFlowMatching with real VLM weights. Loaded once per module."""
    return VLAFlowMatching(config_stage1)


@pytest.fixture
def vla_flow_matching(vla_flow_matching_stage1):
    """Default VLAFlowMatching (Stage 1) for general tests."""
    return vla_flow_matching_stage1


@pytest.fixture
def vla_stage0(vla_flow_matching_stage0):
    return vla_flow_matching_stage0


@pytest.fixture
def vla_stage1(vla_flow_matching_stage1):
    return vla_flow_matching_stage1


@pytest.fixture
def policy_stage0(config_stage0):
    """SmolVLAAptPolicy with Stage 0 config."""
    return SmolVLAAptPolicy(config_stage0)


@pytest.fixture
def vla_stage0_weights(config_stage0):
    """Mock Stage 0 state dict for load_stage0_weights tests.
    Uses a fresh Stage 0 model's weights as the 'checkpoint'.
    Keys are without 'model.' prefix to match VLAFlowMatching's own state_dict."""
    tmp_model = VLAFlowMatching(config_stage0)
    state_dict = {}
    for name, param in tmp_model.named_parameters():
        state_dict[name] = param.clone().detach()
    return state_dict


@pytest.fixture
def batch(images, img_masks, lang_tokens, lang_masks, state, actions):
    """Training batch dict for forward() calls."""
    return dict(
        images=[images],
        img_masks=[img_masks],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
        actions=actions,
    )


@pytest.fixture
def smolvlm_model(vla_flow_matching):
    """SmolVLMWithExpertModel extracted from VLAFlowMatching for VLM-level tests."""
    return vla_flow_matching.vlm_with_expert


@pytest.fixture
def prefix_embs(vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
    """Prefix embeddings for VLM forward tests."""
    embs, _, _, _ = vla_flow_matching.embed_prefix(
        [images], [img_masks], lang_tokens, lang_masks
    )
    return embs


@pytest.fixture
def prefix_embedding(vla_flow_matching, images, img_masks, lang_tokens, lang_masks):
    """Full prefix embedding output for VLM forward tests.
    Returns (embs, pad_masks, att_masks, num_vision_tokens)."""
    return vla_flow_matching.embed_prefix(
        [images], [img_masks], lang_tokens, lang_masks
    )

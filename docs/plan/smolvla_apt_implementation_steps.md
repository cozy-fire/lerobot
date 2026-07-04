# SmolVLA-APT 实施步骤（逐行改动指南）

> 对照文档：`docs/plan/smolvla_apt_design.md`
> 参考实现：`D:\CodeProject\APT\apt\action_expert.py`、`apt/vla.py`

---

## 文件 1：`src/lerobot/policies/smolvla_apt/configuration_smolvla_apt.py`

### 1.1 新增配置字段 + 删除废弃字段（第 92 行后，`attention_mode` 行删除）

```python
# === 删除以下废弃字段 ===
# attention_mode: str = "cross_attn"       # 删除：不再需要 cross/self 模式
# self_attn_every_n_layers: int = 2        # 删除：不再需要 cross/self 层交错
# train_state_proj: bool = True            # 删除：state_proj 已移除（state 迁移到 AE）

# === 新增：APT 2-Stage 预训练配置 ===
# 训练阶段：0 = VA Prior (Vision-Action, 语言屏蔽), 1 = VLA (语言注入)
train_stage: int = 0

# VLM highway 采样间隔。16 层 VLM / 2 = 8 个 highways
vl_highway_interval: int = 2

# Gate fusion 初始化值（0 = sigmoid(0) = 0.5，VA/VLM 各半）
gate_fusion_init: float = 0.0
# ===================================
```

> **注意**：`num_ae_layers` 不从 config 读取，而是由 `vl_highway_interval` 和 `num_vlm_layers` 自动计算：
> `num_highways = num_vlm_layers // vl_highway_interval` → 16//2 = 8 → `num_ae_layers = num_highways = 8`

### 1.2 在 `__post_init__` 中新增验证（第 118 行前插入）

```python
        if self.train_stage not in (0, 1):
            raise ValueError(f"train_stage must be 0 or 1, got {self.train_stage}")
```

---

## 文件 2：`src/lerobot/policies/smolvla_apt/smolvlm_with_expert.py`

### 2.1 新增：VLM 中间层 hidden_state 提取

`forward()` 保持现有双流接口 `inputs_embeds=[prefix, suffix]`（`SmolVLMWithExpertModel` 原生设计），不做破坏性修改。新增 highway 收集逻辑：

在第 426 行的 `for layer_idx in range(num_layers):` 循环**之前**，新增：

```python
        # === 新增：收集 VLM highway（中间层 hidden_states） ===
        interval = getattr(self, 'vl_highway_interval', 2)
        vl_highway_indices = list(range(0, num_layers, interval))
        collected_hidden_states = []  # 存储 VLM 侧的 hidden_states
        # =====================================================
```

在每层 attention+MLP 完成后（原第 457 行 `outputs_embeds = []` 之前），新增 highway 收集：

```python
            # === 新增：收集 VLM hidden_state（仅 VLM 侧，不含 expert） ===
            if layer_idx in vl_highway_indices:
                with torch.no_grad():  # VLM 冻结，不保留 grad
                    vlm_hidden = outputs_embeds[0].detach() if outputs_embeds[0] is not None else None
                collected_hidden_states.append(vlm_hidden)
            # =============================================================
```

返回值在原 `return outputs_embeds, past_key_values` 基础上新增第三项：

```python
        return outputs_embeds, past_key_values, collected_hidden_states
```

> **注意**：`past_key_values` 在 VLM forward 中仍保留（不做破坏性删除），但调用方（`VLAFlowMatching`）忽略它，只用 `collected_hidden_states`。

### 2.2 删除 `past_key_values` 相关代码

**策略**：保守处理。`SmolVLMWithExpertModel.forward()` 保持现有接口不变（`past_key_values`、`use_cache`、`fill_kv_cache` 参数保留不删），仅新增 `collected_hidden_states` 返回值。`VLAFlowMatching` 调用时只传 `inputs_embeds=[prefix_embs]`（单元素），忽略 `past_key_values` 返回值。

**`VLAFlowMatching.denoise_step()` 方法（原 modeling_smolvla_apt.py 第 863-896 行）**：
- 整体删除。推理时 AE 在外部 `HybridAttentionLayers` 中处理，不再需要逐步查 KV cache。

**`SmolVLMWithExpertModel.forward_cross_attn_layer()` 和 expert k_proj/v_proj**：
- 保留不删（避免破坏 `__init__` 的权重加载路径），但 `VLAFlowMatching` 侧不再调用 cross-attn 路径。

**`lm_expert` 层组**：
- 整体保留，参考原始 SmolVLA 的处理方式，不做删除或修改。

### 2.3 VLM forward 调用方式总结

外部 `VLAFlowMatching` 调用 VLM 时只传 prefix，忽略 `past_key_values` 返回值，只取 `collected_hidden_states`：

```python
_, _, vl_highways = self.vlm_with_expert.forward(
    attention_mask=att_2d_masks,
    position_ids=position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs],          # 单元素 list，不传 suffix
    use_cache=False,
    fill_kv_cache=False,
)
# vl_highways: list[Tensor] — 每采样层的 VLM hidden state
```

---

## 文件 3：`src/lerobot/policies/smolvla_apt/modeling_smolvla_apt.py`

这是改动最大的文件。按模块逐一说明。

### 3.1 新增类：`GateFusionBlock`（在 `SmolVLAAptPolicy` 类之前插入，约第 222 行前）

```python
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
        """
        Args:
            vl_tokens: (B, L_vl, D) — AE self-attention output for VL portion
            vlm_highway: (B, L_vl, D) — projected VLM hidden state at this layer
            layer_idx: which layer index for gate lookup
        Returns:
            vl_tokens: (B, L_vl, D) — gated fusion result
        """
        gi = self.gate[layer_idx].sigmoid()  # (D,)
        vl_tokens = vl_tokens * gi + vlm_highway * (1 - gi)
        return vl_tokens
```

### 3.2 新增类：`HybridAttentionLayers`（在 `GateFusionBlock` 之后）

```python
class HybridAttentionLayers(nn.Module):
    """
    Interleaved self-attention layers for VLA.
    
    Layer 数量 = num_highways（VLM highway 数）。
    
    Stage 0: ceil(num_layers/2) 个 VA 层（奇数 gate 索引），语言被 dilated_mask 阻断。
    Stage 1: num_layers 个层，偶数=语言注入（完整 mask+RoPE），奇数=VA（dilated mask+正弦PE）。
    
    每个 AE 层都有独立的 gate + highway 注入（无遗漏）。
    
    Reference: APT apt/action_expert.py:124-172
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,       # = num_highways（如 8）
        train_stage: int,
        head_dim: int,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.train_stage = train_stage
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 构建层列表
        if train_stage == 0:
            # Stage 0: 仅 VA 层，数量 = ceil(num_layers/2)
            # 例：num_layers=8 → 4 层，使用 odd gate 索引 [1,3,5,7]
            va_num_layers = (num_layers + 1) // 2
            self.num_active_layers = va_num_layers
            pe_types = [("sinusoidal", None)] * va_num_layers
        else:
            # Stage 1: 全部 num_layers 层，交错排列
            # 例：num_layers=8 → 8 层，偶数=RoPE，奇数=正弦PE
            if num_layers % 2 != 0:
                raise ValueError(f"Stage 1 requires even num_layers, got {num_layers}")
            self.num_active_layers = num_layers
            pe_types = []
            for i in range(num_layers):
                if i % 2 == 0:
                    pe_types.append((None, "rope"))          # 语言注入层
                else:
                    pe_types.append(("sinusoidal", None))    # vision-action 层
        
        self.layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, head_dim, pe_type=pe_types[i])
            for i in range(self.num_active_layers)
        ])
        
        # Gate fusion: num_layers 个 gate（Stage 1 全量）
        self.gate_fusion = GateFusionBlock(num_layers, hidden_dim, gate_init)
    
    def forward(
        self,
        x: Tensor,                    # (B, L_vl + L_state + L_action, D)
        attention_mask: Tensor,       # (B, L, L) — 完整 mask（偶数层用）
        dilated_mask: Tensor | None,  # (B, L, L) — 阻断语言交叉注意力的 mask（奇数层/Stage 0 用）
        vla_split_sizes: tuple[int, int],  # (len_vl, len_state_plus_action)
        vl_highways: list[Tensor],    # 每层一个 projected VLM hidden state
        position_ids: Tensor,         # (B, L)
    ) -> Tensor:
        """
        Returns:
            x: (B, L, D) — updated tokens after all layers
        """
        for i, layer in enumerate(self.layers):
            # 确定该层的 mask：偶数=完整，奇数/Stage 0=dilated
            if self.train_stage == 0:
                mask = dilated_mask if dilated_mask is not None else attention_mask
            elif i % 2 == 0:
                mask = attention_mask          # 语言注入层：完整 mask
            else:
                mask = dilated_mask if dilated_mask is not None else attention_mask  # VA 层：dilated mask
            
            x = layer(x, attention_mask=mask, position_ids=position_ids)
            
            # Gate Fusion：将 VLM highway 注入 VL tokens
            # gate 和 highway 使用同一索引 j（一对一绑定，参考 APT）
            j = (2 * i + 1) if self.train_stage == 0 else i
            if j < len(vl_highways) and vl_highways[j] is not None:
                len_vl, len_rest = vla_split_sizes
                vl, rest = x.split([len_vl, len_rest], dim=1)
                vl = self.gate_fusion(vl, vl_highways[j], j)
                x = torch.cat([vl, rest], dim=1)
        
        return x
```

### 3.3 新增辅助类：`AttentionBlock`（参考 APT `apt/layers/attn.py:77-113`）

APT 的 `SelfAttentionBlock` 结构：`RMSNorm → Self-Attn(+RoPE/正弦PE) → residual → RMSNorm → SwiGLU-FFN → residual`。

与 APT 的差异：
- APT 用 AdaRMSNorm + FiLM 做 timestep 调制；SmolVLA-APT 改为 **Concat+MLP** 方式（`embed_ae_tokens` 中已处理），AttentionBlock 用**普通 RMSNorm**
- PRoPE → 正弦 PE（SmolVLA 无相机外参）
- RoPE 保留给语言注入层

```python
class AttentionBlock(nn.Module):
    """
    Self-attention + SwiGLU-FFN block.
    
    Reference: APT apt/layers/attn.py:77-113 (SelfAttentionBlock)
               APT apt/layers/attn.py:24-34 (FFN / SwiGLU)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        pe_type: tuple = (None, "rope"),  # (vision_pe, lang_pe): "rope"|"sinusoidal"|None
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.pe_type = pe_type
        self.num_heads = num_heads      # 用于 mask expand
        
        # Norm 1: pre-attention (RMSNorm, 无 FiLM)
        self.norm1 = nn.RMSNorm(hidden_dim)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Norm 2: pre-FFN (RMSNorm, 无 FiLM)
        self.norm2 = nn.RMSNorm(hidden_dim)
        
        # FFN: SwiGLU
        ffn_hidden = hidden_dim * ffn_expansion
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, hidden_dim, bias=False)
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,        # (B, L, L) BoolTensor
        position_ids: Tensor | None = None,
    ) -> Tensor:
        residual = x
        x = self.norm1(x)
        # nn.MultiheadAttention 不接受 (B, L, L) mask，需 expand 到 (B*num_heads, L, L)
        B, L, _ = attention_mask.shape
        mask_expanded = attention_mask.unsqueeze(1).expand(B, self.num_heads, L, L).reshape(B * self.num_heads, L, L)
        x, _ = self.self_attn(x, x, x, attn_mask=mask_expanded)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        x = x + residual
        return x
```

> **删除**：`AdaRMSNorm` 类不再需要（timestep 调制统一在 `embed_ae_tokens` 中通过 Concat+MLP 完成）。

### 3.4 修改 `VLAFlowMatching.__init__()`（第 555-593 行）

改动点汇总：

1. **移除 `state_proj`（第 569-571 行）**：state 不再进入 VLM prefix
2. **新增 `ae_state_proj`**：state 投影到 AE hidden_dim
3. **新增 `vl_highway_proj`**：VLM hidden states → AE hidden_dim 的投影层组
4. **新增 `hybrid_attn_layers`**：替代原有的 expert cross-attention
5. **新增 `vl0_proj`**：VLM inputs_embeds → AE hidden_dim

**具体替换**：

删除原有（第 559-582 行）的 `vlm_with_expert` 构建中 `attention_mode` 相关逻辑，
修改 `SmolVLMWithExpertModel` 调用，新增 `return_highways=True` 参数（见文件 2）。

```python
    def __init__(self, config: SmolVLAAptConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.train_stage = config.train_stage

        # === VLM Backbone（仅 Image + Language，不含 State） ===
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode="self_attn",    # 不再需要 cross_attn
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=-1,   # 废弃：gate fusion 中不再需要 self/cross 层交错
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        # 设置 highway 采样间隔
        self.vlm_with_expert.vl_highway_interval = self.config.vl_highway_interval
        
        ae_hidden_dim = self.vlm_with_expert.expert_hidden_size  # AE 隐层维度
        
        # === VLM → AE 投影 ===
        # vl0: VLM pre-transformer embeddings (prefix_embs) → AE hidden_dim
        # 参考 APT action_expert.py:242: proj_input(inputs_embeds)
        vlm_hidden_dim = self.vlm_with_expert.config.text_config.hidden_size
        self.vl0_proj = nn.Linear(vlm_hidden_dim, ae_hidden_dim)
        
        # vl_highway_proj: VLM 每层 hidden_state → AE hidden_dim
        num_highways = len(list(range(0, self.config.num_vlm_layers, self.config.vl_highway_interval)))
        self.vl_highway_proj = nn.ModuleList([
            nn.Linear(vlm_hidden_dim, ae_hidden_dim) for _ in range(num_highways)
        ])
        
        # === State + Action 投影（都在 AE 中处理） ===
        self.ae_state_proj = nn.Linear(self.config.max_state_dim, ae_hidden_dim)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, ae_hidden_dim)
        self.action_out_proj = nn.Linear(ae_hidden_dim, self.config.max_action_dim)
        
        # 时间嵌入 MLP（融合 timestep + action）
        self.action_time_mlp_in = nn.Linear(ae_hidden_dim * 2, ae_hidden_dim)
        self.action_time_mlp_out = nn.Linear(ae_hidden_dim, ae_hidden_dim)
        
        # === Hybrid Attention Layers（替代原有 expert cross-attn） ===
        # AE 层数 = num_highways（每个 highway 对应一个 AE 层）
        # 16 VLM 层 / interval=2 = 8 highways → 8 AE 层
        # Stage 0: ceil(8/2)=4 VA 层 → 使用 odd gate 索引 [1,3,5,7]
        # Stage 1: 8 层（4 even + 4 odd）→ 所有 gate 索引 [0..7]
        num_ae_layers = num_highways
        self.hybrid_attn_layers = HybridAttentionLayers(
            hidden_dim=ae_hidden_dim,
            num_heads=self.vlm_with_expert.config.text_config.num_attention_heads,
            num_layers=num_ae_layers,
            train_stage=self.train_stage,
            head_dim=self.vlm_with_expert.config.text_config.head_dim,
            gate_init=self.config.gate_fusion_init,
        )
        
        self.set_requires_grad()
        
        # === 保留原有 token 相关属性 ===
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor
```

#### `set_requires_grad()` 重写

原方法只控制 `self.state_proj`，需重写适配新模块。AE 全参数可训练，VLM 由 `freeze_vision_encoder + train_expert_only` 冻结：

```python
def set_requires_grad(self):
    for m in [self.ae_state_proj, self.vl0_proj, self.action_in_proj,
              self.action_out_proj, self.action_time_mlp_in, self.action_time_mlp_out,
              self.hybrid_attn_layers, *self.vl_highway_proj]:
        for p in m.parameters():
            p.requires_grad = True
```

### 3.5 修改 `embed_prefix()`（第 617-709 行）—— 移除 state

**删除**第 684-697 行（state embedding 部分）：

```python
        # 删除以下行：
        # state_emb = self.state_proj(state)
        # state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        # embs.append(state_emb)
        # ...
        # att_masks += [1] * (states_seq_len)
```

同时修改函数签名：**移除 `state` 参数，新增返回 `num_vision_tokens`**：

```python
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            embs: (B, L, D) — VLM 输入嵌入
            pad_masks: (B, L)
            att_masks: (B, L)
            num_vision_tokens: int — vision 部分的 token 数量（language 之前）
                用于 prepare_attention_mask 区分 V/L 边界
        """
```

并在拼接 language 之前记录边界（原第 680 行前插入）：

```python
        # 记录 vision/language 边界（language 拼接前 embs 中已有的 token 数即为 vision）
        num_vision_tokens = sum(e.shape[1] for e in embs)  # 累加所有 image token 数
        
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        ...  # 原有代码继续
```

最后 return 改为：

```python
        return embs, pad_masks, att_masks, num_vision_tokens
```

### 3.6 修改 `embed_suffix()` → 重命名为 `embed_ae_tokens()`（第 711-752 行）

新函数签名和逻辑：

```python
    def embed_ae_tokens(self, noisy_actions, timestep, state):
        """
        将 state + noisy_actions + timestep 嵌入为 AE tokens。
        State 不再进入 VLM，而是直接进入 AE。
        
        Args:
            noisy_actions: (B, chunk_size, max_action_dim)
            timestep: (B,) — flow matching time
            state: (B, max_state_dim)
        Returns:
            ae_embs: (B, 1 + chunk_size, ae_hidden_dim) — [state_token, action_tokens...]
            ae_pad_masks: (B, 1 + chunk_size)
            ae_att_masks: (B, 1 + chunk_size) — 全部设为 1（AE 内部双向注意力）
        """
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        
        # 1. State token
        state_emb = self.ae_state_proj(state)  # (B, hidden_dim)
        state_emb = state_emb.unsqueeze(1)     # (B, 1, hidden_dim)
        
        # 2. Action tokens（与 timestep 融合）
        action_emb = self.action_in_proj(noisy_actions)
        dtype = action_emb.dtype
        
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size,
            self.config.min_period, self.config.max_period, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        
        # 3. 拼接 [State, Actions]
        ae_embs = torch.cat([state_emb, action_time_emb], dim=1)  # (B, 1+chunk, D)
        
        # 4. Mask: 全部可见
        ae_pad_masks = torch.ones(bsize, ae_embs.shape[1], dtype=torch.bool, device=device)
        ae_att_masks = torch.zeros(bsize, ae_embs.shape[1], dtype=torch.long, device=device)
        
        return ae_embs, ae_pad_masks, ae_att_masks
```

### 3.7 新增 `prepare_attention_mask()`（在 `VLAFlowMatching` 类内）

Vision/Language 边界由 `embed_prefix()` 返回的 `num_vision_tokens` 确定，无需从 VLM 内部获取。

```python
    def prepare_attention_mask(
        self,
        vl_mask: Tensor,            # (B, L_vl) — VLM attention mask
        ae_mask: Tensor,            # (B, L_ae) — AE token mask
        num_vision_tokens: int,     # vision token 数量（language 之前）
        train_stage: int,
    ) -> tuple[Tensor, Tensor]:
        """
        构建 self-attention mask for [Vision + Language + AE] 序列。
        
        参考 APT action_expert.py:18-71 (prepare_attention_mask)
        
        Token 布局：[Vision(0..Nv) | Language(Nv..Nv+Nl) | AE(Nv+Nl..)]
        
        Returns:
            full_mask: (B, L, L) — 完整 mask，所有 token 双向可见
            dilated_mask: (B, L, L) — 阻断 Language↔Vision 和 Language↔AE 交叉注意力
        """
        bsize = vl_mask.shape[0]
        device = vl_mask.device
        L_vl = vl_mask.shape[1]
        L_ae = ae_mask.shape[1]
        L = L_vl + L_ae
        
        # modality_type: 0=PAD, 1=VISION, 2=LANGUAGE, 3=ACTION
        modality = torch.full((bsize, L), 3, dtype=torch.long, device=device)
        modality[:, :num_vision_tokens] = 1                     # VISION
        modality[:, num_vision_tokens:L_vl] = 2                 # LANGUAGE
        # L_vl 之后 → 已经是 ACTION (3)
        
        # 完整 mask（所有 token 互相可见）
        full_mask = torch.ones(bsize, L, L, dtype=torch.bool, device=device)
        
        # Dilated mask：阻断语言与 vision/action 的交叉注意力
        # Language token (modality=2) 不能看到 Vision (1) 和 Action (3)
        # Vision (1) 和 Action (3) 不能看到 Language (2)
        is_lang = modality == 2   # (B, L)
        # 扩展为 2D mask
        lang_block_mask = is_lang[:, :, None] | is_lang[:, None, :]  # (B, L, L): lang→any 或 any→lang
        
        dilated_mask = full_mask.clone()
        if train_stage == 0:
            # Stage 0: 完全阻断 language → 所有语言交叉注意力被屏蔽
            dilated_mask = dilated_mask & ~lang_block_mask
        else:
            # Stage 1: 奇数层用 dilated_mask
            dilated_mask = dilated_mask & ~lang_block_mask
        
        return full_mask, dilated_mask
```

### 3.8 修改 `VLAFlowMatching.forward()`（第 754-790 行）

完全重写训练 forward：

```python
    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass with gate fusion."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Step 1: VLM 前向（仅 Image + Language，不含 State）
        prefix_embs, prefix_pad_masks, prefix_att_masks, num_vision_tokens = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # VLM forward：保持 [prefix, suffix] 双流接口，suffix=None
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, _, vl_highways = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        # vl_highways = list of (B, L_prefix, vlm_hidden_dim) per sampled layer
        
        # Step 2: vl0 从 prefix_embs（pre-transformer）投影，非 VLM 输出
        # 参考 APT action_expert.py:242 — proj_input(inputs_embeds)
        vl0 = self.vl0_proj(prefix_embs)  # (B, L_vl, ae_hidden_dim)
        vl0_pad_mask = prefix_pad_masks
        
        # 投影 highways（含越界保护 S11）
        vl_highways_proj = []
        for i, hw in enumerate(vl_highways):
            if i < len(self.vl_highway_proj) and hw is not None:
                vl_highways_proj.append(self.vl_highway_proj[i](hw))
            else:
                vl_highways_proj.append(None)
        
        # Step 3: 嵌入 AE tokens（State + Noisy Actions）
        ae_embs, ae_pad_masks, ae_att_masks = self.embed_ae_tokens(x_t, time, state)
        
        # Step 4: 拼接 VL + AE tokens
        x = torch.cat([vl0, ae_embs], dim=1)  # (B, L_vl + L_ae, D)
        x_pad_masks = torch.cat([vl0_pad_mask, ae_pad_masks], dim=1)
        x_att_masks = torch.cat([prefix_att_masks, ae_att_masks], dim=1)
        
        # 构建 mask
        full_mask, dilated_mask = self.prepare_attention_mask(
            vl0_pad_mask, ae_pad_masks, num_vision_tokens, self.train_stage
        )
        
        # Step 5: Hybrid Attention（self-attn over V+L+S+A）
        position_ids_full = torch.cumsum(x_pad_masks, dim=1) - 1
        x = self.hybrid_attn_layers(
            x=x,
            attention_mask=full_mask,
            dilated_mask=dilated_mask,
            vla_split_sizes=(vl0.shape[1], ae_embs.shape[1]),
            vl_highways=vl_highways_proj,
            position_ids=position_ids_full,
        )
        
        # Step 6: 提取 Action tokens，输出预测
        suffix_out = x[:, -self.config.chunk_size:]  # 取 action 部分
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses
```

### 3.9 修改 `sample_actions()`（第 792-861 行）—— 推理路径

同理重写推理前向：

```python
    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device
        
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        # Step 1: VLM 前向（一次，结果在去噪循环中复用）
        prefix_embs, prefix_pad_masks, prefix_att_masks, num_vision_tokens = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, _, vl_highways = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        
        # vl0 从 prefix_embs（pre-transformer）投影，非 VLM 输出
        vl0 = self.vl0_proj(prefix_embs)
        vl0_pad_mask = prefix_pad_masks
        
        vl_highways_proj = []
        for i, hw in enumerate(vl_highways):
            if i < len(self.vl_highway_proj) and hw is not None:
                vl_highways_proj.append(self.vl_highway_proj[i](hw))
            else:
                vl_highways_proj.append(None)
        
        # Step 2: 去噪循环
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            
            # 嵌入 AE tokens
            ae_embs, ae_pad_masks, ae_att_masks = self.embed_ae_tokens(x_t, time_tensor, state)
            
            # 拼接
            x = torch.cat([vl0, ae_embs], dim=1)
            full_mask, dilated_mask = self.prepare_attention_mask(
                vl0_pad_mask, ae_pad_masks, num_vision_tokens, self.train_stage
            )
            
            x = self.hybrid_attn_layers(
                x=x,
                attention_mask=full_mask,
                dilated_mask=dilated_mask,
                vla_split_sizes=(vl0.shape[1], ae_embs.shape[1]),
                vl_highways=vl_highways_proj,
                position_ids=torch.cumsum(torch.cat([vl0_pad_mask, ae_pad_masks], dim=1), dim=1) - 1,
            )
            
            suffix_out = x[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            
            x_t = x_t + dt * v_t
        
        return x_t
```

> **RTC 兼容**：原 `sample_actions` 的 RTC 分支传入 `denoise_step_partial_call` 闭包来获取 `v_t`，该闭包依赖 `past_key_values`。新架构中改为依赖预计算的 `vl0` + `vl_highways_proj`（在去噪循环外部计算一次，循环内复用）。RTC `denoise_step` 接口不变。

### 3.10 新增 `load_stage0_weights()`（在 `SmolVLAAptPolicy` 类中，约第 245 行后）

```python
    def load_stage0_weights(self, stage0_state_dict: dict):
        """
        从 Stage 0 checkpoint 加载权重到 Stage 1 模型。
        
        参考 APT vla.py:101-143 (load_from_pretrain)
        
        Stage 0 Layer i → Stage 1 Layer 2*i + 1（奇数层）
        其他参数（投影层等）直接复制。
        """
        if self.config.train_stage != 1:
            raise ValueError("load_stage0_weights only works when train_stage=1")
        
        model_state = self.model.state_dict()
        
        # 映射 hybrid attention 层
        # Stage 0 有 N/2 层，Stage 1 有 N 层
        s0_layer_idx = 0
        for s1_layer_idx in range(len(self.model.hybrid_attn_layers.layers)):
            if s1_layer_idx % 2 == 1:  # 奇数层 = 继承 Stage 0
                for name, _ in self.model.hybrid_attn_layers.layers[s1_layer_idx].named_parameters():
                    k1 = f"model.hybrid_attn_layers.layers.{s1_layer_idx}.{name}"
                    k0 = f"model.hybrid_attn_layers.layers.{s0_layer_idx}.{name}"
                    if k0 in stage0_state_dict:
                        model_state[k1] = stage0_state_dict[k0]
                s0_layer_idx += 1
        
        # 直接复制投影层（不含 gate — APT 中 gate 在 Stage 1 全新零初始化）
        for name, param in self.model.named_parameters():
            if "hybrid_attn_layers" not in name and "gate_fusion" not in name:
                full_name = f"model.{name}"
                if full_name in stage0_state_dict:
                    model_state[full_name] = stage0_state_dict[full_name]
        
        self.model.load_state_dict(model_state, strict=False)
```

### 3.11 修改 `SmolVLAAptPolicy.__init__()`（第 229-245 行）

添加 Stage 0→1 权重加载逻辑：

```python
    def __init__(self, config: SmolVLAAptConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        
        # === 新增：Stage 0→1 权重加载 ===
        if config.train_stage == 1 and config.pretrained_path:
            import os
            import torch as _torch
            ckpt_path = os.path.join(config.pretrained_path, "model.safetensors")
            if os.path.exists(ckpt_path):
                from safetensors.torch import load_file
                stage0_sd = load_file(ckpt_path)
                self.load_stage0_weights(stage0_sd)
        
        self.reset()
```

### 3.12 删除 PEFT 相关方法（在 `SmolVLAAptPolicy` 类中）

原 SmolVLA 从 HF 下载全参数预训练权重后，用户微调时走 PEFT/LoRA 路径（只训练 `lm_expert` Q/V 投影 + 几个投影层）。新架构中：
- `lm_expert` 不再使用（AE 移到 `hybrid_attn_layers`）
- AE 仅有 ~8 层 AttentionBlock，参数量小，全参数微调即可
- VLM 通过 `freeze_vision_encoder=True` + `train_expert_only=True` 冻结

**删除以下两个方法**（原 modeling_smolvla.py 中）：

```python
# 删除 _get_default_peft_targets()
# 删除 _validate_peft_config()
```

不需要新增任何替代逻辑——LeRobot 框架检测到没有 `_get_default_peft_targets` 时会默认全参数训练（`set_requires_grad` 已控制 VLM 冻结）。

---

## 文件 4：`src/lerobot/policies/smolvla_apt/processor_smolvla_apt.py`

### 4.1 无需大改（第 39-103 行）

processor 基本保持不变，因为：
- State 的归一化仍在 `NormalizerProcessorStep` 中处理（特征级别）
- 只是 State 数据不再被 VLM 的 `embed_prefix` 消费，而是被 AE 的 `embed_ae_tokens` 消费

`make_smolvla_apt_pre_post_processors` 保持不变即可。

---

## 文件 5：`src/lerobot/policies/factory.py`（已注册，无需改动）

已在之前的步骤中完成 `smolvla_apt` 注册。无需额外改动。

---

## 实施顺序建议

| 序号 | 文件 | 改动范围 | 风险 |
|------|------|----------|------|
| 1 | `configuration_smolvla_apt.py` | 新增 4 个字段 | 低 |
| 2 | `smolvlm_with_expert.py` | forward 返回 highways | 中 |
| 3 | `modeling_smolvla_apt.py` — 辅助类 | 新增 3 个类（~150 行） | 低 |
| 4 | `modeling_smolvla_apt.py` — `__init__` | 重写构造函数 | 中 |
| 5 | `modeling_smolvla_apt.py` — `embed_prefix` | 删除 state 嵌入（~15 行删除） | 低 |
| 6 | `modeling_smolvla_apt.py` — `embed_ae_tokens` | 重写 suffix 嵌入（~40 行） | 中 |
| 7 | `modeling_smolvla_apt.py` — `forward` | 重写训练/推理 forward（~60 行各） | 高 |
| 8 | `modeling_smolvla_apt.py` — `load_stage0_weights` | 新增（~30 行） | 中 |
| 9 | `modeling_smolvla_apt.py` — `SmolVLAAptPolicy.__init__` | 新增 Stage 权重加载（~10 行） | 低 |
| 10 | `processor_smolvla_apt.py` | 无需改动 | — |
| 11 | `modeling_smolvla_apt.py` — 删除 `denoise_step` | 删除旧 cross-attn 推理方法 | 低 |
| 12 | `smolvlm_with_expert.py` — 删除 `past_key_values` | 清理 KV cache 相关代码 | 中 |

---

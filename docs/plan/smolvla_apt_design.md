# SmolVLA-APT 实现计划

> 目标：参考 APT（Action Expert Pretraining）论文 [arXiv:2606.12366](https://arxiv.org/abs/2606.12366)，为 SmolVLA 引入 2-Stage 预训练方案，验证增加 VA 先验对泛化性能的提升。

---

## 1. 架构对比总览

### 当前 SmolVLA 数据流
```
┌─────────────────────────────────────────────────────────┐
│  VLM Backbone (SmolVLM)                                  │
│  [Image Embed] + [Language Embed] + [State Embed]        │
│       ↓                                                  │
│  Prefix KV Cache ────Cross-Attn──→ Action Expert         │
│                                      ↑                   │
│                                 [Noisy Actions]          │
│                                      ↓                   │
│                              [Predicted Actions]         │
└─────────────────────────────────────────────────────────┘
```

### 目标 SmolVLA-APT 数据流
```
┌─────────────────────────────────────────────────────────┐
│  VLM Backbone (SmolVLM, Frozen)                          │
│  [Image Embed] + [Language Embed]                        │
│       ↓                                                  │
│  inputs_embeds + vl_highways (per-layer hidden states)   │
│       ↓                                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Action Expert (Trainable)                       │    │
│  │  [V tokens + L tokens] + [State] + [Noisy Acts]  │    │
│  │       ↓                                          │    │
│  │  HybridAttentionLayers (self-attn V+L+S+A)       │    │
│  │       ↓                                          │    │
│  │  Gate Fusion: vl ⊙ σ(g) + highway ⊙ (1-σ(g))    │    │
│  │       ↓                                          │    │
│  │  [Predicted Actions]                             │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 核心改动：Gate Fusion 替代 Cross-Attention

### 2.1 当前 Cross-Attention 机制（删除）
- VLM 先跑 prefix 生成 KV Cache，AE 每层通过 cross-attention 查 VLM 的 K/V
- `past_key_values` 在 denoising 循环中逐步维护 → 全部删除

### 2.2 新 Gate Fusion 机制（参考 APT `apt/action_expert.py:143-171`）
- VLM 只跑一次 prefix，返回 `collected_hidden_states`（vl_highways），不再维护 KV cache
- AE 做 self-attention over `[V+L] + [State] + [NoisyActions]`，每层用独立 gate 做逐通道融合
- `vl0` 从 `prefix_embs`（pre-transformer embedding）投影，而非 VLM 最后一层输出（APT 风格，非 GR00T 风格）
- timestep 仅通过 Concat+MLP 注入 action token（`embed_ae_tokens`），不额外使用 FiLM/AdaRMSNorm

```python
gate = nn.Parameter(torch.zeros(num_layers, hdim))  # sigmoid(0)=0.5，VA/VLM 均等
gi = gate[layer_idx].sigmoid()
vl = vl * gi + vl_highways[layer_idx] * (1 - gi)
```

### 2.3 VLM Highway 提取
- `SmolVLMWithExpertModel.forward()` 新增返回 `collected_hidden_states`（保留 `past_key_values` 接口兼容但调用方忽略）
- 每隔 2 层采样：16 层 VLM → 8 个 highway，每个通过独立 Linear 投影到 `ae_hidden_dim`
- AE 层数 = highway 数（8 层）：Stage 0 有 4 个 VA 层，Stage 1 有 8 层（4 even + 4 odd）
- 偶数层 = 语言注入（RoPE + 完整 mask），奇数层 = VA（正弦 PE + dilated mask 阻断语言）

---

## 3. State（本体感知）迁移

### 3.1 当前：State 在 VLM Backbone
- `embed_prefix()` 中 `state_proj(state)` → 拼入 VLM prefix（第 686-701 行）
- State token 参与 VLM 的 self-attention，影响 KV Cache

### 3.2 目标：State 在 Action Expert
- 从 `embed_prefix()` 移除 state 嵌入
- 在 Action Expert 中新增 `state_proj`（映射到 expert_hidden_size）
- State token 拼入 AE 的 self-attention 序列：`[V+L] + [State] + [Noisy Actions]`
- 不附带历史动作（仅当前 state + 当前 noisy actions）

---

## 4. 2-Stage 预训练方案

### 4.1 Stage 0: VA Prior（Vision-Action 先验）
- **train_stage = 0**
- **VLM 完全冻结**（`freeze_vision_encoder=True`, `train_expert_only=True`）
- **语言被完全屏蔽**：使用 `causal_mask_dilated`（阻断 L↔V、L↔A 所有交叉注意力）
- Action Expert 仅使用 vision-action 层（每 2 层取 1 层，索引为奇数）
- 位置编码：仅使用 pose-residual PE（或简化版正弦 PE）
- **目标**：在纯 Vision-Action 数据上学到鲁棒的视觉运动先验

### 4.2 Stage 1: VLA Likelihood（语言注入）
- **train_stage = 1**
- 从 Stage 0 加载权重：Stage 0 的第 i 层 → Stage 1 的第 2i+1 层（奇数层）
- 新增偶数层（语言注入层），随机初始化
- 偶数层使用 RoPE 位置编码 + 完整因果注意力
- 奇数层仍使用 dilated mask（语言与 vision/action 隔离）
- Gate fusion 作用于所有层
- **目标**：在保持 VA 先验的同时，将语言条件注入模型

### 4.3 权重加载（Stage 0 → Stage 1）
参考 `apt/vla.py:101-143` 的 `load_from_pretrain()`：
```
Stage 0 Layer 0 → Stage 1 Layer 1
Stage 0 Layer 1 → Stage 1 Layer 3
Stage 0 Layer 2 → Stage 1 Layer 5
...
```
其余参数（投影层、norm 层等）直接复制。

---

## 5. 需修改的文件清单

### 5.1 `configuration_smolvla_apt.py`
- **新增**：`train_stage`（0/1）、`vl_highway_interval`（默认 2）、`gate_fusion_init`（默认 0.0）
- **删除**：`attention_mode`、`self_attn_every_n_layers`、`train_state_proj`（gate fusion 架构中废弃）
- `num_ae_layers` 不从 config 读取，由 `num_vlm_layers // vl_highway_interval` 自动计算

### 5.2 `modeling_smolvla_apt.py`
- **新增 3 个类**：`GateFusionBlock`、`HybridAttentionLayers`、`AttentionBlock`（RMSNorm + MHA + SwiGLU，无 FiLM）
- **`VLAFlowMatching.__init__`**：删除 `state_proj`，新增 `ae_state_proj`/`vl0_proj`/`vl_highway_proj`/`hybrid_attn_layers`，重写 `set_requires_grad`
- **`embed_prefix()`**：移除 state 参数，新增返回 `num_vision_tokens`（V/L 边界）用于 dilated mask
- **`embed_suffix()` → `embed_ae_tokens()`**：state + noisy actions + timestep 嵌入为 AE tokens
- **新增 `prepare_attention_mask()`**：基于 modality 构建 dilated mask（阻断 L↔V、L↔A）
- **`forward()`**：VLM 只跑一次 `inputs_embeds=[prefix_embs]`，AE 通过 hybrid attention + gate fusion 输出
- **`sample_actions()`**：vl0 + vl_highways 在去噪循环外预计算，循环内复用
- **删除 `denoise_step()`**：不再需要逐步查 KV cache
- **`SmolVLAAptPolicy`**：新增 `load_stage0_weights()`（Stage 0 Layer i → Stage 1 Layer 2i+1），删除 PEFT 方法

### 5.3 `smolvlm_with_expert.py`
- `forward()` 新增返回 `collected_hidden_states`（vl_highways），保留现有接口不变
- `lm_expert`/`forward_cross_attn_layer`/k_proj/v_proj 均保留（不破坏权重加载路径）

### 5.4 `processor_smolvla_apt.py`
- 无需改动（state 归一化仍在特征级别处理，只是消费方从 VLM 变为 AE）

### 5.5 `factory.py`
- 已完成 `smolvla_apt` 注册，无需额外改动

---

## 6. 训练流程

### Stage 0 训练
```bash
lerobot-train \
  --policy.type=smolvla_apt \
  --policy.train_stage=0 \
  --policy.freeze_vision_encoder=True \
  --policy.train_expert_only=True \
  --dataset.repo_id=<robot_datasets> \
  --batch_size=64 \
  --steps=100000
```

### Stage 1 训练
```bash
lerobot-train \
  --policy.type=smolvla_apt \
  --policy.train_stage=1 \
  --policy.pretrained_path=<stage0_checkpoint> \
  --dataset.repo_id=<target_task_dataset> \
  --batch_size=64 \
  --steps=50000
```

---

## 7. 实施步骤

1. **修改 `configuration_smolvla_apt.py`** — 新增 `train_stage`、`num_ae_layers` 等配置项
2. **修改 `smolvlm_with_expert.py`** — 返回 VLM 中间层 hidden_states（vl_highways）
3. **重构 `VLAFlowMatching`** — 实现 Gate Fusion + Hybrid Attention：
   - 新增 `GateFusionBlock` 类
   - 新增 `HybridAttentionLayers` 类（支持 interleaved layers + dual masks）
   - 移除 cross-attention，改为 self-attention over [VL + State + NoisyActions]
   - State 从 prefix 迁移到 AE
4. **修改 `SmolVLAAptPolicy`** — 支持 train_stage 参数 + Stage 0→1 权重加载
5. **调整 processor** — state 归一化移入 AE
6. **集成测试** — 确保 Stage 0 / Stage 1 都能正常训练

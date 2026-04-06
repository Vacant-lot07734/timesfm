# TimesFM 2.5 Torch 版架构与推理流程详解

本文面向代码阅读，重点解释以下文件如何协同工作：

- `src/timesfm/timesfm_2p5/timesfm_2p5_torch.py`
- `src/timesfm/timesfm_2p5/timesfm_2p5_base.py`
- `src/timesfm/torch/dense.py`
- `src/timesfm/torch/transformer.py`
- `src/timesfm/torch/util.py`
- `src/timesfm/configs.py`

目标是回答 4 个问题：

1. 这个模型的架构到底是什么。
2. 输入一条时间序列以后，推理时数据是怎么流动的。
3. 每一步张量形状如何变化。
4. 代码里哪些 Python 语法是理解这份实现的关键。

---

## 1. 先看整体：这个文件做了两层封装

`src/timesfm/timesfm_2p5/timesfm_2p5_torch.py` 里其实有两层对象：

### 1.1 底层真正的 PyTorch 模型

类：`TimesFM_2p5_200M_torch_module`

它负责：

- 定义网络层
- 加载 checkpoint
- 实现 `forward()`
- 实现真正的 patch-level decoding 逻辑 `decode()`

这一层更像“纯模型本体”。

### 1.2 面向用户的模型包装器

类：`TimesFM_2p5_200M_torch`

它负责：

- 从 Hugging Face 下载或加载本地权重
- 处理 `ForecastConfig`
- 把 Python / NumPy 输入转换成 torch tensor
- 做推理前后的额外逻辑
  - 输入标准化
  - flip invariance
  - continuous quantile head
  - quantile crossing 修复
  - 非负约束

这一层更像“推理服务接口层”。

可以把它理解成：

```text
用户输入
  -> TimesFM_2p5_200M_torch
      -> TimesFM_2p5_200M_torch_module
          -> tokenizer + 20 层 Transformer + 两个输出头
```

---

## 2. 配置决定了模型的骨架

TimesFM 2.5 的核心超参数不在 `timesfm_2p5_torch.py` 里硬编码，而是在
`src/timesfm/timesfm_2p5/timesfm_2p5_base.py` 的
`TimesFM_2p5_200M_Definition` 中定义。

关键参数如下：

- `context_limit = 16384`
- `input_patch_len = 32`
- `output_patch_len = 128`
- `output_quantile_len = 1024`
- `num_layers = 20`
- `model_dims = 1280`
- `num_heads = 16`
- `head_dim = 1280 / 16 = 80`
- `quantiles = [0.1, ..., 0.9]`
- `decode_index = 5`

这几个值直接决定了后面所有张量形状。

---

## 3. 模型架构：它不是逐点预测，而是“32 输入 patch -> 128 输出 patch”

这是理解 TimesFM 的第一关键点。

### 3.1 输入 patch

模型不是把每个时间点当作一个 token，而是把连续值序列切成长度为 `p=32`
的 patch。

例如上下文长度 `context=256`：

- 原始输入形状：`[B, 256]`
- 切 patch 后：`[B, 8, 32]`

这里：

- `B` 是 batch size
- `8 = 256 / 32` 是 patch 数

### 3.2 输出 patch

每个 patch token 并不只预测一个未来点，而是预测一个长度为 `o=128` 的未来块。

这意味着：

- 一个输入 token 表示一段 32 点历史
- 一个输出 token 预测未来 128 点

这也是为什么它能比“1 token -> 1 token”的自回归更高效。

### 3.3 输出通道

`self.q = len(quantiles) + 1 = 10`

10 个通道分别是：

- 第 0 个：mean
- 第 1 到 9 个：q10, q20, ..., q90

其中 `decode_index = 5` 对应 q50，也就是中位数。模型在自回归解码时，用的是第 5 个通道作为下一轮输入。

---

## 4. `__init__()`：模型层是怎么搭起来的

`TimesFM_2p5_200M_torch_module.__init__()` 的结构很简单，但每个成员都很关键。

### 4.1 几个重要缩写

代码里用了大量单字母缩写：

- `p = 32`：输入 patch 长度
- `o = 128`：输出 patch 长度
- `os = 1024`：连续 quantile head 的输出长度
- `m = o // p = 4`
- `x = 20`：Transformer 层数
- `h = 16`：attention 头数
- `md = 1280`：model dims
- `hd = 80`：每个 head 的维度
- `q = 10`：输出通道数
- `aridx = 5`：自回归使用第 5 个通道，即 q50

其中 `m = 4` 很重要，因为：

```text
128 个输出点 = 4 个长度为 32 的输入 patch
```

所以模型每次生成完 128 点预测后，会把这 128 点再切回 4 个 patch，作为下一轮输入。

### 4.2 tokenizer

```python
self.tokenizer = dense.ResidualBlock(self.config.tokenizer)
```

这里用的是 `dense.ResidualBlock`，配置来自：

- `input_dims = 64`
- `hidden_dims = 1280`
- `output_dims = 1280`

为什么输入维度是 64？

因为在 `forward()` 里做了这一步：

```python
tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
```

也就是把：

- 32 个数值输入
- 32 个 mask 位

拼到一起，得到 64 维输入。

所以 tokenizer 不是单纯只看数值，它同时看到“哪些位置是 padding”。

### 4.3 Transformer 主干

```python
self.stacked_xf = nn.ModuleList(
  [transformer.Transformer(...) for _ in range(self.x)]
)
```

这是 20 层 decoder-only Transformer。

每层结构在 `src/timesfm/torch/transformer.py` 的 `Transformer` 类中：

- pre-attn RMSNorm
- MultiHeadAttention
- post-attn RMSNorm + residual
- pre-ff RMSNorm
- FFN
- post-ff RMSNorm + residual

也就是非常标准的 Pre-Norm Transformer Block。

### 4.4 两个输出头

```python
self.output_projection_point = dense.ResidualBlock(...)
self.output_projection_quantiles = dense.ResidualBlock(...)
```

两个头分别输出：

- `output_projection_point.output_dims = 1280 = 128 * 10`
- `output_projection_quantiles.output_dims = 10240 = 1024 * 10`

注意这里名字有一点迷惑：

- `point` 头实际上并不只是“点预测”
- 它输出的是长度 128 的未来 patch，每个时间点带 10 个通道
- 后面会 reshape 成 `[B, *, 128, 10]`

第二个头是更长 horizon 的连续 quantile spread 头，后面会 reshape 成
`[B, *, 1024, 10]`。

---

## 5. `ResidualBlock` 到底做了什么

定义在 `src/timesfm/torch/dense.py`：

```python
output = output_layer(activation(hidden_layer(x))) + residual_layer(x)
```

这不是 Transformer block，而是一个简单的 MLP 残差块：

- 一条主支路：`Linear -> Activation -> Linear`
- 一条残差支路：`Linear`
- 两者相加

如果输入是：

- `x.shape = [B, N, D_in]`

输出就是：

- `shape = [B, N, D_out]`

在 tokenizer 中：

- 输入 `[B, N_patch, 64]`
- 输出 `[B, N_patch, 1280]`

在 point head 中：

- 输入 `[B, N_patch, 1280]`
- 输出 `[B, N_patch, 1280]`

在 continuous quantile head 中：

- 输入 `[B, N_patch, 1280]`
- 输出 `[B, N_patch, 10240]`

---

## 6. Transformer 内部：attention 张量如何变化

重点看 `src/timesfm/torch/transformer.py` 的 `MultiHeadAttention.forward()`。

假设输入：

- `inputs_q.shape = [B, N, 1280]`

### 6.1 线性投影成 Q/K/V

如果 `fuse_qkv=True`：

```python
qkv = self.qkv_proj(inputs_q)
query, key, value = torch.chunk(qkv, 3, dim=-1)
```

那么：

- `qkv.shape = [B, N, 3840]`
- `query.shape = [B, N, 1280]`
- `key.shape = [B, N, 1280]`
- `value.shape = [B, N, 1280]`

然后 reshape：

- `[B, N, 1280] -> [B, N, 16, 80]`

差别 1：参数组织形式不同
-   非融合：Q、K、V 各自有一个独立线性层，对应三个权重矩阵
-   融合：仍然本质上对应 Q、K、V 三组参数，但把它们在实现上合并成一个大矩阵，一次输出 `3 * hidden_dim`，再切分

差别 2：计算实现方式不同
-   非融合：需要做三次线性变换
-   融合：只需做一次更大的线性变换，减少访存与调度开销，通常性能更好


### 6.2 Rotary Position Embedding

RoPE 作用在：

- `query: [B, N, 16, 80]`
- `key: [B, N, 16, 80]`

注意它没有对 `value` 做位置编码。

### 6.3 attention mask

`make_attn_mask()` 生成的是一个 causal mask，同时把完全被 padding 的 key/value 位置屏蔽掉。

输出 mask 的可理解形状是：

- `[B, 1, Q, K]`

这里：

- `Q` 是 query 长度
- `K` 是 key/value 长度

### 6.4 attention 输出

注意 `_torch_dot_product_attention()` 先把张量从：

- `[B, L, H, D]`

转成：

- `[B, H, L, D]`

因为 PyTorch fused attention kernel 要求 head 维在前面。

attention 输出再转回：

- `[B, L, H, D]`

最后 reshape 回：

- `[B, L, 1280]`

然后再过 `self.out`。

---

## 7. `forward()`：一次前向传播的数据流

`TimesFM_2p5_200M_torch_module.forward()` 输入的是“已经 patch 化并归一化后的序列”。

假设：

- `inputs.shape = [B, N_patch, 32]`
- `masks.shape = [B, N_patch, 32]`

### 第一步：把数值和 mask 拼接给 tokenizer

```python
tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
```

形状变化：

- `[B, N, 32] + [B, N, 32] -> [B, N, 64]`

### 第二步：tokenizer 投影到 model dims

```python
input_embeddings = self.tokenizer(tokenizer_inputs)
```

形状：

- `[B, N, 64] -> [B, N, 1280]`

### 第三步：通过 20 层 Transformer

```python
output_embeddings = input_embeddings
for layer in self.stacked_xf:
    output_embeddings, new_cache = layer(...)
```

每层都保持形状不变：

- `[B, N, 1280] -> [B, N, 1280]`

### 第四步：两个输出头

```python
output_ts = self.output_projection_point(output_embeddings)
output_quantile_spread = self.output_projection_quantiles(output_embeddings)
```

输出形状：

- `output_ts: [B, N, 1280]`
- `output_quantile_spread: [B, N, 10240]`

### 第五步：返回中间结果

`forward()` 返回了 4 份张量：

- `input_embeddings`
- `output_embeddings`
- `output_ts`
- `output_quantile_spread`

以及新的 `decode_caches`

这说明这个 `forward()` 不只是服务最终推理，也方便调试和分析中间层表示。

---

## 8. `decode()`：真正的推理主流程

`decode()` 是整个文件最重要的函数。

它负责把一个完整上下文变成最终 forecast。

输入：

- `inputs.shape = [B, context]`
- `masks.shape = [B, context]`

其中：

- `context` 已经在 `compile()` 之前被对齐成 `32` 的倍数

### 8.1 计算 decode 步数

```python
num_decode_steps = (horizon - 1) // self.o
```

这里 `self.o = 128`。

这意味着：

- 第一个 128 点预测来自 prefill 的最后一个 token
- 之后每多 128 点，就多做一次自回归 decode

例如：

- `horizon = 1~128` -> `num_decode_steps = 0`
- `horizon = 129~256` -> `num_decode_steps = 1`
- `horizon = 257~384` -> `num_decode_steps = 2`

### 8.2 先把输入切成 patch

```python
patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
patched_masks = torch.reshape(masks, (batch_size, -1, self.p))
```

形状变化：

- `[B, context] -> [B, N_in, 32]`

其中：

- `N_in = context / 32`

### 8.3 计算每个 patch 的运行统计量

这段逻辑用到了 `util.update_running_stats()`。

它逐 patch 累积：

- `n`：已看到的有效元素数
- `mu`：均值
- `sigma`：标准差

最终堆叠成：

- `context_mu.shape = [B, N_in]`
- `context_sigma.shape = [B, N_in]`

这里的意义是：

- 第 `i` 个 patch 的归一化，不是用全局固定统计量
- 而是用“看到第 i 个 patch 为止”的历史运行统计量

这是一种带时序因果性的 ReVIN。

### 8.4 分配 KV cache

每一层 Transformer 都会创建一个 `DecodeCache`：

- `next_index.shape = [B]`
- `num_masked.shape = [B]`
- `key.shape = [B, cache_size, 16, 80]`
- `value.shape = [B, cache_size, 16, 80]`

其中：

```text
cache_size = num_input_patches + num_decode_steps * m
```

因为每次自回归扩展 128 点，其实会被重新切成 `m=4` 个输入 patch。

### 8.5 prefill：先把真实历史喂进去

```python
normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
(_, _, normed_outputs, normed_quantile_spread), decode_caches = self(...)
```

张量形状：

- `patched_inputs: [B, N_in, 32]`
- `context_mu/context_sigma: [B, N_in]`
- `normed_inputs: [B, N_in, 32]`
- `normed_outputs: [B, N_in, 1280]`
- `normed_quantile_spread: [B, N_in, 10240]`

这里有两个关键点：

#### 关键点 1：为什么还要把 masked 位置置 0

因为左 padding 之后，某个 patch 可能部分有效、部分无效。

例如：

- patch 长度 32
- 前 28 个位置是 padding
- 后 4 个位置是真实值

这时：

- patch 级别这个 token 仍然需要保留
- 但 patch 内被 mask 的位置必须清零

所以模型通过两种方式知道 padding：

- 数值位置被清零
- mask 位作为额外 32 维输入拼给 tokenizer

#### 关键点 2：patch 级 mask 为什么用 `masks[..., -1]`

在 `forward()` 里传给 Transformer 的是：

```python
masks[..., -1]
```

也就是每个 patch 最后一个位置的 mask。

这并不是在说“patch 内只有最后一个点重要”，而是在做 patch 级别的“是否可参与 attention”的近似标记：

- 如果一个 patch 的最后一个位置也是 mask，说明这个 patch 全是 padding
- 如果最后一个位置有效，说明这个 patch 至少有真实内容，应该保留

patch 内更细粒度的 mask 信息已经通过 tokenizer 输入表达了。

### 8.6 prefill 输出 reshape 成未来 patch

```python
renormed_outputs = torch.reshape(
    revin(normed_outputs, context_mu, context_sigma, reverse=True),
    (batch_size, -1, self.o, self.q),
)
```

形状变化：

- `normed_outputs: [B, N_in, 1280]`
- 先反归一化，形状不变
- 再 reshape 成：
  - `[B, N_in, 128, 10]`

这说明：

- 每个输入 patch token
- 对应输出一个长度 128 的未来块
- 每个未来点有 10 个通道

第二个头同理：

```python
renormed_quantile_spread.shape = [B, N_in, 1024, 10]
```

随后只取最后一个 token 对应的长 horizon quantile 输出：

```python
renormed_quantile_spread = ...[:, -1, ...]
```

最终得到：

- `[B, 1024, 10]`

### 8.7 取最后一个 token 的 q50 作为下一轮输入

```python
last_renormed_output = renormed_outputs[:, -1, :, self.aridx]
```

其中：

- `[:, -1, :, :]`：取最后一个上下文 patch 的预测
- `self.aridx = 5`：取第 5 个通道，也就是 q50

结果形状：

- `[B, 128]`

这 128 个点就是“当前最可信的未来块”，后面会被切成 4 个 patch 继续喂回去。

---

## 9. 自回归解码：为什么每次要把 128 点再切回 4 个 patch

自回归部分：

```python
new_patched_input = torch.reshape(last_renormed_output, (batch_size, self.m, self.p))
```

因为：

- `128 = 4 * 32`

所以：

- `[B, 128] -> [B, 4, 32]`

这一步非常关键。

模型训练时认识的是“32 长度 patch token”，不是“128 长度 patch token”。
所以每次生成 128 点以后，必须重新切成 4 个长度 32 的 patch，才能继续喂给同一个 tokenizer + Transformer。

### 9.1 新一轮统计量

新生成的 4 个 patch 并不是直接原样送进去，而是继续用累积统计量更新：

- `new_mu.shape = [B, 4]`
- `new_sigma.shape = [B, 4]`

然后做 ReVIN：

- `new_normed_input.shape = [B, 4, 32]`

### 9.2 新一轮 Transformer

```python
(_, _, new_normed_output, _), decode_caches = self(
    new_normed_input, new_mask, decode_caches
)
```

这里的关键是：

- 这次不再是全量 prefill
- 而是带着之前的 `decode_caches` 继续增量解码

输出：

- `new_normed_output.shape = [B, 4, 1280]`

反归一化并 reshape 后：

- `new_renormed_output.shape = [B, 4, 128, 10]`

然后取最后一个 patch 的预测：

- `new_renormed_output[:, -1, ...].shape = [B, 128, 10]`

并追加到 `ar_outputs`

### 9.3 自回归循环的输出

如果有 `S = num_decode_steps` 轮循环，那么：

- `ar_outputs` 里每个元素是 `[B, 128, 10]`
- `torch.stack(ar_outputs, dim=1)` 后变成：
  - `[B, S, 128, 10]`

---

## 10. `decode()` 最终返回什么

`decode()` 返回 3 个对象：

### 10.1 `renormed_outputs`

形状：

- `[B, N_in, 128, 10]`

含义：

- 对每个输入 patch token 的 128 点 future patch 预测

通常真正用于 forecast 的是最后一个输入 patch 对应的输出。

### 10.2 `renormed_quantile_spread`

形状：

- `[B, 1024, 10]`

含义：

- 最后一个 token 的 continuous quantile spread head 输出

### 10.3 `ar_renormed_outputs`

形状：

- `[B, S, 128, 10]`

或者 `None`

含义：

- 额外自回归扩展出来的输出块

---

## 11. `compile()`：真正给用户返回 forecast 结果的包装层

`TimesFM_2p5_200M_torch.compile()` 不是编译神经网络本体，而是构造一个闭包 `_compiled_decode` 并保存到 `self.compiled_decode`。

所以它实际上做了两件事：

1. 校验并修正 `ForecastConfig`
2. 生成一个最终推理入口函数

### 11.1 配置修正

如果用户传入：

- `max_context` 不是 32 的倍数
- `max_horizon` 不是 128 的倍数

会自动向上取整。

例如：

- `max_context=1000 -> 1024`
- `max_horizon=250 -> 256`

这是因为模型内部所有 patch 操作都依赖固定块大小。

### 11.2 `_compiled_decode()` 的输入

```python
inputs: list[np.ndarray] -> torch.Tensor [B, C]
masks:  list[np.ndarray] -> torch.Tensor [B, C]
```

其中：

- `C = max_context`

如果启用了 `ForecastConfig.normalize_inputs=True`，这里还会先做一层整序列标准化：

- `mu.shape = [B, 1]`
- `sigma.shape = [B, 1]`

注意：

这里和 `decode()` 内部的 patch-level ReVIN 是两层不同的归一化：

- 外层：整条输入序列级别
- 内层：因果 patch 级别运行统计量

### 11.3 拼接 prefill 与 AR 输出

`decode()` 返回之后：

```python
to_cat = [pf_outputs[:, -1, ...]]
if ar_outputs is not None:
    to_cat.append(ar_outputs.reshape(batch_size, -1, self.model.q))
full_forecast = torch.cat(to_cat, dim=1)
```

形状变化：

- `pf_outputs[:, -1, ...]`：`[B, 128, 10]`
- `ar_outputs.reshape(...)`：`[B, S*128, 10]`
- 拼接后：
  - `[B, 128 + S*128, 10]`

如果这个长度超过实际 `horizon`，后面会裁掉：

```python
full_forecast = full_forecast[:, :horizon, :]
```

### 11.4 flip invariance

如果 `fc.force_flip_invariance=True`，代码会再跑一遍：

- 输入取负：`-inputs`
- 再 decode 一次
- 对 quantile 做翻转
- 再与正向结果合并

直觉上就是强制满足近似关系：

```text
f(-x) ≈ -f(x)
```

为什么 quantile 需要翻转？

因为对负数取相反数时：

- q10 会变成 q90
- q20 会变成 q80
- ...

所以用了：

```python
x[..., :1]
torch.flip(x[..., 1:], dims=(-1,))
```

保持 mean 通道不动，分位数通道反序。

### 11.5 continuous quantile head

如果启用 `fc.use_continuous_quantile_head=True`，代码会把：

- `quantile_spreads`

重新映射到：

- `full_forecast[:, :, quantile_index]`

实现方式是：

```python
spread_q - spread_q50 + median
```

也就是说：

- 第 5 个通道 q50 作为中心值
- 其他 quantile 用 continuous head 的相对偏移量构造出来

### 11.6 return_backcast

如果 `fc.return_backcast=True`：

```python
full_backcast = pf_outputs[:, :-1, : self.model.p, :].reshape(batch_size, -1, self.model.q)
full_forecast = torch.cat([full_backcast, full_forecast], dim=1)
```

这里会把历史 reconstruction 也一起拼进返回结果。

这主要服务于 XReg / covariate 路径，而不是普通 forecast。

### 11.7 quantile crossing 修复

如果启用 `fc.fix_quantile_crossing=True`：

- 先从 q40 往 q10 回扫，保证低分位单调递减
- 再从 q60 往 q90 正扫，保证高分位单调递增

这是一种后处理，而不是模型内部约束。

### 11.8 最终返回

```python
return full_forecast[..., 5], full_forecast
```

也就是：

- 点预测：第 5 个通道 q50，形状 `[B, horizon]`
- 全量输出：形状 `[B, horizon, 10]`

---

## 12. 推理数据流总览

下面用统一符号总结一遍。

设：

- `B`：batch size
- `C`：context length
- `N = C / 32`：输入 patch 数
- `H`：目标 horizon
- `S = floor((H - 1) / 128)`：自回归步数

### 12.1 进入模型前

来自 `timesfm_2p5_base.py`：

- 原始输入：`list[np.ndarray]`
- pad / truncate 后：
  - `inputs: [B, C]`
  - `masks: [B, C]`

### 12.2 `decode()` prefill

- `patched_inputs: [B, N, 32]`
- `patched_masks: [B, N, 32]`
- `context_mu/context_sigma: [B, N]`
- `tokenizer_inputs: [B, N, 64]`
- `input_embeddings: [B, N, 1280]`
- `output_embeddings: [B, N, 1280]`
- `output_ts: [B, N, 1280]`
- reshape 后：
  - `renormed_outputs: [B, N, 128, 10]`
- quantile head：
  - `renormed_quantile_spread: [B, 1024, 10]`

### 12.3 AR decode

每轮：

- `last_renormed_output: [B, 128]`
- `new_patched_input: [B, 4, 32]`
- `new_normed_output: [B, 4, 1280]`
- `new_renormed_output: [B, 4, 128, 10]`
- 追加：
  - `[B, 128, 10]`

循环结束：

- `ar_outputs: [B, S, 128, 10]`

### 12.4 最终拼接

- `pf_outputs[:, -1, ...]: [B, 128, 10]`
- `ar_outputs.reshape(...): [B, S*128, 10]`
- `full_forecast: [B, 128 + S*128, 10]`
- 裁剪到 horizon：
  - `[B, H, 10]`

最终返回：

- 点预测：`[B, H]`
- 全量分位数：`[B, H, 10]`

---

## 13. 这份实现里最关键的 Python 语法

这一节不讲通用 Python 教程，只讲本文件里不理解就很难顺着读下去的语法。

### 13.1 类型注解里的联合类型

例如：

```python
decode_caches: list[util.DecodeCache] | None = None
```

意思是：

- `decode_caches` 要么是 `list[DecodeCache]`
- 要么是 `None`

`|` 是 Python 3.10 之后的联合类型写法，等价于旧写法：

```python
Optional[list[util.DecodeCache]]
```

### 13.2 类定义里给基类传关键字参数

例如：

```python
class TimesFM_2p5_200M_torch(
  timesfm_2p5_base.TimesFM_2p5,
  PyTorchModelHubMixin,
  library_name="timesfm",
  repo_url="https://github.com/google-research/timesfm",
  ...
):
```

这不是普通继承写法。

这里的关键点是：

- Python 允许在 `class` 语句里给基类系统传关键字参数
- 这些参数会进入基类的 `__init_subclass__()` 流程

在这里，它主要是给 `PyTorchModelHubMixin` 提供 Hugging Face Hub 相关元数据。

### 13.3 `@classmethod`

例如：

```python
@classmethod
def _from_pretrained(cls, ...):
```

这表示：

- 方法绑定在类上，而不是实例上
- 第一个参数不是 `self`，而是 `cls`

所以它可以在里面创建实例：

```python
instance = cls(...)
```

### 13.4 `*` 后面的 keyword-only 参数

例如：

```python
def _from_pretrained(
    cls,
    *,
    model_id: str = DEFAULT_REPO_ID,
    revision: Optional[str],
    ...
)
```

`*` 的意思是：

- 后面的参数必须用关键字传入
- 不能靠位置来传

这样可以避免像 `from_pretrained("repo", None, "/tmp", True, False, ...)`
这种难读且容易传错的调用。

### 13.5 海象运算符 `:=`

文件里多次出现：

```python
if (w := len(value)) >= context:
```

意思是：

1. 先把 `len(value)` 赋给 `w`
2. 再判断 `w >= context`

这叫 walrus operator。

还有一个更“怪”的例子在 `util.update_running_stats()`：

```python
return (w := (new_n, new_mu, new_sigma), w)
```

按 Python 语义，这行会先把：

```python
(new_n, new_mu, new_sigma)
```

赋给变量 `w`，然后返回：

```python
(w, w)
```

也就是返回两份完全相同的元组。

所以调用方才会写：

```python
(n, mu, sigma), _ = util.update_running_stats(...)
```

第二个返回值其实没提供额外信息，更多是接口遗留。

### 13.6 多重解包

例如：

```python
(_, _, normed_outputs, normed_quantile_spread), decode_caches = self(...)
```

这里做了两层解包：

第一层：

- `self(...)` 返回 `(tuple_of_outputs, decode_caches)`

第二层：

- `tuple_of_outputs` 里又有 4 个元素

因此可以一次性解出：

- 前两个中间值丢掉，用 `_`
- 保留 `normed_outputs`
- 保留 `normed_quantile_spread`
- 以及新的 `decode_caches`

### 13.7 `...` 省略号切片

例如：

```python
x[..., :1]
x[..., 1:]
masks[..., -1]
```

`...` 表示“前面所有维度”。

比如：

- `x.shape = [B, H, Q]`
- `x[..., :1]` 等价于 `x[:, :, :1]`

这种写法特别适合高维张量，避免手写一长串冒号。

### 13.8 list comprehension

例如：

```python
self.stacked_xf = nn.ModuleList(
  [transformer.Transformer(...) for _ in range(self.x)]
)
```

这是一种“用一行生成列表”的写法。

这里生成的是：

- 20 个 `Transformer(...)`
- 再放入 `nn.ModuleList`

`_` 表示循环变量本身不需要被使用。

### 13.9 上下文管理器 `with`

例如：

```python
with torch.no_grad():
```

表示在这个代码块里：

- 不记录梯度
- 节省显存
- 用于纯推理

这是 PyTorch 推理代码的标准写法。

### 13.10 `self = torch.compile(self)` 这行的 Python 语义

这是一个值得特别注意的点。

在 `load_checkpoint()` 里：

```python
if torch_compile:
    self = torch.compile(self)
```

从 Python 语义看，这只是把函数内部的局部变量 `self` 重新绑定到
`torch.compile(self)` 的返回值。

它不会自动把外部持有的 `instance.model` 替换掉。

也就是说，如果这里依赖的是“返回一个新的 compiled module 对象”，那么仅靠这句局部赋值，并不能把编译后的模块真正挂回实例字段。

这不是语法错误，但阅读时要明确：

- 这是“局部变量重绑定”
- 不是“原对象原地改写”的语义

---

## 14. 阅读这份实现时最容易忽略的 5 个点

### 14.1 它有两层归一化

- 外层：`compile()` 里的整序列归一化
- 内层：`decode()` 里的因果 ReVIN

如果只注意到其中一层，很容易误判数值流。

### 14.2 tokenizer 的输入不是 32，而是 64

因为把 patch 值和 patch mask 拼起来了。

### 14.3 point head 名字有误导性

它输出的不是单点，而是：

- `128 * 10 = 1280`

也就是一个完整 future patch。

### 14.4 自回归不是 1 步 1 点，而是 1 步 128 点

但为了继续喂回模型，会把这 128 点切成 4 个长度 32 的 patch。

### 14.5 最终点预测不是 mean，而是 q50

返回值第一项：

```python
full_forecast[..., 5]
```

也就是中位数，而不是第 0 通道的 mean。

---

## 15. 一句话总结

这份实现的核心思想可以概括成：

> 用长度 32 的 patch 把连续值序列编码成 token，经 20 层 decoder-only
> Transformer 建模后，每个 token 直接输出一个长度 128 的未来块；推理时先用
> 最后一个历史 token 生成第一块未来，再把这块未来切回 4 个 patch 做增量
> 自回归，直到覆盖目标 horizon。

如果只记一条张量主线，可以记这个：

```text
[B, C]
-> [B, C/32, 32]
-> [B, C/32, 64]
-> [B, C/32, 1280]
-> [B, C/32, 128, 10]
-> [B, H, 10]
-> [B, H]
```

其中中间自回归阶段还会反复经过：

```text
[B, 128]
-> [B, 4, 32]
-> [B, 4, 1280]
-> [B, 4, 128, 10]
```

这就是 TimesFM 2.5 Torch 版推理代码最核心的数据流。

---

## 16. `xreg` 是什么，它的典型逻辑是什么

### 16.1 `xreg` 的本质

这里的 `xreg` 可以理解成：

- `exogenous regression`
- 或者更直白一点：带外生变量的线性回归

它不是 Transformer 里面的一部分，不是 attention 模块，也不是额外的神经网络层。

在 TimesFM 2.5 这份实现里，`xreg` 是包在模型外侧的一层“协变量校正器”：

- TimesFM 负责看目标序列本身的历史形态
- XReg 负责利用额外协变量做线性解释和修正

相关入口在：

- `src/timesfm/timesfm_2p5/timesfm_2p5_base.py` 的 `forecast_with_covariates()`
- `src/timesfm/utils/xreg_lib.py`

所以要先有 TimesFM 正常的 `forecast()` 能力，才谈得上 `forecast_with_covariates()`。

### 16.2 它支持哪些 covariate

`forecast_with_covariates()` 支持四类协变量：

- `dynamic_numerical_covariates`
  - 随时间变化的数值特征
  - 例如温度、价格、促销力度
- `dynamic_categorical_covariates`
  - 随时间变化的类别特征
  - 例如星期几、节假日类型
- `static_numerical_covariates`
  - 每条序列固定不变的数值特征
- `static_categorical_covariates`
  - 每条序列固定不变的类别特征

如果是按金融例子理解：

- dynamic numerical：技术指标、已知日历数值特征
- dynamic categorical：交易日星期、月份、session 类型
- static categorical：股票代码、交易所、行业

### 16.3 为什么它要求未来 covariates 已知

在 `forecast_with_covariates()` 里，代码会根据 dynamic covariate 的长度推断 horizon：

```python
test_lens.append(len(dynamic_covariate) - input_len)
```

这说明：

- 传入的 dynamic covariate 长度必须是 `context + horizon`
- 也就是历史段和未来段都要有值

所以这里的 `xreg` 只能用在“未来协变量可知”的场景。

例如：

- 可以：`day_of_week`
- 可以：`month_of_year`
- 可以：已知的促销计划
- 不可以：未来真实成交量
- 不可以：未来真实 high/low

这也是为什么在金融场景里，`xreg` 更适合作为“时间类 / 已知计划类”辅助信息，而不是把未来 OHLCVA 真值直接喂进去。

### 16.4 `xreg` 的数据准备逻辑

`forecast_with_covariates()` 会先做几件事情：

#### 第一步：决定 train / test 长度

它把：

- 上下文段叫 `train`
- 未来 horizon 段叫 `test`

这是线性回归里的命名，不是深度学习里的 train/test dataset 划分。

#### 第二步：把 dynamic covariate 切成 context 部分和 horizon 部分

这一段代码很关键：

```python
train_covariates[covariate_name].append(
    covariate_value[(input_len - train_len) : input_len]
)
test_covariates[covariate_name].append(covariate_value[input_len:])
```

也就是：

- 历史协变量进入线性模型拟合
- 未来协变量进入线性模型外推

#### 第三步：构建设计矩阵

`xreg_lib.BatchedInContextXRegBase.create_covariate_matrix()` 会把所有协变量摊平，拼成标准线性回归矩阵：

- 数值特征直接拼列
- 类别特征 one-hot 编码
- 静态特征会按每条序列长度 repeat 展开
- 最后可以加 intercept 列

最终得到：

- `flat_targets`: `[总样本数]`
- `x_train`: `[总训练样本数, 特征维]`
- `x_test`: `[总未来样本数, 特征维]`

### 16.5 `xreg` 实际上拟合的是一个线性模型

在 `BatchedInContextXRegLinear.fit()` 中，最终是解这个问题：

```text
beta_hat = (X^T X + ridge I)^(-1) X^T y
```

代码里用的是 JAX 的伪逆实现：

```python
beta_hat = pinv(X^T X + ridge I) @ X^T @ y
```

所以：

- `ridge=0` 时近似普通最小二乘
- `ridge>0` 时是 ridge regression

这也是为什么 `xreg` 本质是“线性校正器”，不是“又叠了一层神经网络”。

### 16.6 两种工作模式

`xreg_mode` 有两种。

#### 模式 A：`timesfm + xreg`

逻辑：

1. 先用 TimesFM 做 forecast
2. 再用协变量去拟合 residual
3. 最后把 residual correction 加回到 TimesFM 输出

代码里对应：

```python
targets = actual_context - timesfm_backcast
```

然后拟合：

```text
residual ~ covariates
```

最后：

```python
new_point_outputs = point_output + xreg
```

适合场景：

- TimesFM 已经能抓住主序列模式
- 协变量主要解释剩余误差

#### 模式 B：`xreg + timesfm`

逻辑：

1. 先用协变量直接拟合 target
2. 得到 context 上的拟合值和未来上的外推值
3. 再把 target 减去协变量解释部分，得到 residual
4. 用 TimesFM 去预测 residual
5. 最后把线性部分加回去

代码里对应：

```python
inputs = target - xreg_on_context
point_outputs = self.forecast(inputs=残差序列)
new_point_outputs = point_outputs + xreg
```

适合场景：

- 协变量能解释较大一部分主信号
- TimesFM 更像 residual forecaster

### 16.7 为什么 `return_backcast=True` 是必须的

在 `forecast_with_covariates()` 开头有这个检查：

```python
elif not self.forecast_config.return_backcast:
    raise ValueError(...)
```

原因是：

- `timesfm + xreg` 模式需要 TimesFM 在 context 上的 reconstruction/backcast
- 只有这样才能计算 residual

如果不返回 backcast，就无法知道：

```text
actual_context - timesfm_context_fit
```

也就无法在 context 段拟合残差线性模型。

### 16.8 一句话总结 `xreg`

`xreg` 不是把 covariate 直接塞进 Transformer 内部，而是：

> 在 TimesFM 外面包了一层“基于上下文拟合、基于未来 covariates 外推”的线性回归器，
> 用于解释或修正目标序列。

---

## 17. Transformer 输入里的 `mask` 是怎么来的，怎么工作的

这个问题非常关键，因为 TimesFM 的 mask 分两层：

1. patch 内的 point-level mask
2. patch 级别的 attention mask

很多人第一次读这份代码时，会把这两层混在一起。

### 17.1 最原始的 mask 从哪里来

最早的 mask 来自 `timesfm_2p5_base.py` 的 `forecast()`：

```python
if len(value) >= context:
    value = value[-context:]
    mask = np.zeros_like(value, dtype=bool)
else:
    mask = np.array([True] * (context - w) + [False] * w)
    value = np.pad(value, (context - w, 0), "constant", constant_values=0.0)
```

这说明：

- `True` 表示 padding 位置
- `False` 表示真实有效位置

而且 padding 发生在左侧，也就是前面补零。

所以输入到 `decode()` 前：

- `inputs.shape = [B, C]`
- `masks.shape = [B, C]`

其中 `C = max_context`

### 17.2 reshape 成 patch 后的 mask

在 `decode()` 里：

```python
patched_masks = torch.reshape(masks, (batch_size, -1, self.p))
```

形状：

- `[B, C] -> [B, N_patch, 32]`

这里每个 patch 里有 32 个布尔值，表示这 32 个位置哪些是 padding。

### 17.3 第一层作用：mask 被拼进 tokenizer 输入

在 `forward()` 中：

```python
tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
```

所以 tokenizer 实际拿到的是：

- 32 个数值
- 32 个 mask 位

也就是：

- `[B, N, 32] + [B, N, 32] -> [B, N, 64]`

这意味着 patch 内部的 mask 信息不是丢了，而是显式地进入了 MLP tokenizer。

因此模型可以知道：

- patch 里哪些点是真值
- 哪些点是左 padding 过来的假值

### 17.4 第二层作用：masked point 会被清零

在 prefill 阶段还有一步：

```python
normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
```

作用是：

- 被 mask 的位置数值强制置 0

所以 point-level mask 有两种表达方式：

1. 数值层面置 0
2. mask 位拼接到 tokenizer 输入

### 17.5 第三层作用：Transformer 只接收 patch 级 mask

在 `forward()` 中，传给每层 Transformer 的不是完整的 `[B, N, 32]` mask，而是：

```python
masks[..., -1]
>>> x = torch.range(0,8).reshape(1,3,3)
>>> x
tensor([[[0., 1., 2.],
         [3., 4., 5.],
         [6., 7., 8.]]])
>>> x[...,-1]
tensor([[2., 5., 8.]])
```

形状：

- `[B, N, 32] -> [B, N]`

这里的语义不是“每个 patch 只看最后一个点”，而是用最后一个点来判断这个 patch token 是否应该被当作纯 padding token。

为什么这个逻辑成立？

因为 padding 是连续地左对齐补出来的。

所以：

- 如果某个 patch 的最后一个位置仍然是 `True`
  - 说明这一整个 patch 都是 padding
- 如果某个 patch 的最后一个位置是 `False`
  - 说明这个 patch 至少已经进入了真实数据区间
  - 即使 patch 前面有部分 padding，也应该保留这个 token

所以：

- `point-level mask` 负责 patch 内部细节
- `patch-level mask` 负责 attention 时是否把整个 token 当作“纯 padding token”

### 17.6 patch mask 在 attention 里怎么工作

在 `MultiHeadAttention.forward()` 中：

```python
num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
```

这表示：

- 每条样本最前面有多少个“整 patch 都是 padding”的 token

然后它被用在两个地方。

#### 用途 A：修正 RoPE 的位置编号

```python
position = arange(n_patches) + next_index - num_masked
```

作用：

- 让真实 token 的位置编号从有效起点开始计数
- 避免前面左 padding 的 patch 把位置编号整体往后推

直觉上就是：

- padding patch 不应该占用真实时间步的位置编号

#### 用途 B：构造 causal attention mask

```python
attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)
```

`make_attn_mask()` 同时做两件事：

1. 保证 causal：未来不能看过去之后的位置
2. 保证 padding patch 不参与 key/value

逻辑是：

```python
q_index >= kv_index
kv_index >= num_all_masked_kv
```

也就是：

- query 只能看当前及以前
- 但最前面的若干个“纯 padding patch”会被完全屏蔽

### 17.7 decode cache 阶段的 mask

当进入增量解码时：

```python
new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)
```

因为自回归阶段送进去的是模型自己刚预测出来的 patch：

- 它们没有 padding
- 所以全部是有效值

这时：

- point-level mask 全 False
- patch-level mask 也全 False

decode cache 里的 `num_masked` 会继承 prefill 阶段的历史 padding 信息，用于维持位置一致性。

### 17.8 一句话总结 mask 机制

这份实现里的 mask 不是单一用途，而是三层并行工作：

1. `forecast()` 构造左 padding mask
2. `decode()` 用 patch 内 mask 做数值清零和 tokenizer 输入增强
3. `Transformer` 用 patch 级 mask 做位置修正和 causal attention 屏蔽

---

## 18. 两个输出头到底在输出什么，分位数是什么意思

### 18.1 两个输出头的结构本身并不复杂

这两个头都只是 `ResidualBlock`：

- `output_projection_point`
- `output_projection_quantiles`

所以从结构上讲，它们都只是“残差 MLP 头”，并不神秘。

复杂的是：

- 它们输出的张量如何被解释
- 它们分别承担什么预测任务

### 18.2 第一个头：`output_projection_point`

配置：

- 输入维度：1280
- 输出维度：1280

因为：

```text
1280 = 128 * 10
```

所以它的输出会被 reshape 成：

```python
[B, N_patch, 128, 10]
```

含义是：

- 对每个输入 patch token
- 输出未来 128 个时间点
- 每个时间点有 10 个通道

这 10 个通道是：

- 均值 mean
- q10: 模型认为未来极大概率（90%）会高于这个数值，它是悲观的下界。
- q50: 模型认为未来有 50% 概率高于此值，50% 概率低于此值。这个通常作为我们平时认知里的核心预测值。（中位数）
- q90: 模型认为未来极大概率（90%）会低于这个数值，它是乐观的上界。
它的意义在于构建置信区间（Confidence Interval）： 通过 q10 和 q90，你能画出一个区间带。

如果今天大盘极其平稳，模型给出的范围可能是 [99, 101]，这就是告诉你它很自信。
如果今天发生了巨大的做空事件波动，模型给出的核心预测 q50 可能还是 100，但区间变成了 [80, 120]。此时即使点预测一样，作为系统你也能通过区间跨度判定当前风险极高、模型不确定性极大，从而采取对冲或降低仓位。

因此这个头本质上是：

> 一个“未来 128 步的基础预测头”

它同时提供：

- mean
- median / quantiles
- 自回归下一轮所需的 q50 通道

### 18.3 第二个头：`output_projection_quantiles`

配置：

- 输入维度：1280
- 输出维度：10240

因为：

```text
10240 = 1024 * 10
```

所以它会被 reshape 成：

```python
[B, N_patch, 1024, 10]
```

但最后只保留最后一个 token 的输出：

```python
[:, -1, ...]
```

得到：

```python
[B, 1024, 10]
```

它的作用不是重新给出一套独立的 point forecast，而是提供一个更长 horizon 的
continuous quantile spread 信息。

### 18.4 为什么叫 quantile spread

看 `compile()` 里的逻辑：

```python
full_forecast[:, :, quantile_index] = (
    quantile_spreads[:, :, quantile_index]
    - quantile_spreads[:, :, 5]
    + full_forecast[:, :, 5]
)
```

这说明第二个头的用法是：

- 它提供各个 quantile 相对 q50 的偏移结构
- `full_forecast[..., 5]` 里的 q50 作为中心值
- 其他 quantile 通过“相对偏移 + q50”重构出来

所以这个头更像：

> 一个专门建模“预测分布形状”的长 horizon 头

而不是简单地再输出一份未来值。

### 18.5 分位数到底是什么意思

这里的分位数是概率预测的标准概念。

例如：

- `q10 = 100`

可以理解成：

- 模型认为未来真实值有大约 10% 的概率低于 100
- 或者有大约 90% 的概率高于 100

同理：

- `q50` 是中位数
- `q90` 表示大约 90% 的概率低于该值

所以：

- `q10 ~ q90` 可以构成一个预测区间
- 区间越宽，表示模型不确定性越高
- 区间越窄，表示模型更确定

### 18.6 mean 和 q50 不是一回事

输出里的第 0 个通道是 mean，第 5 个通道是 q50。

两者区别：

- mean：概率分布的均值
- q50：概率分布的中位数

如果分布是对称的，它们可能接近。
如果分布偏斜，它们会不同。

这也是为什么最终返回点预测时，代码选择的是：

```python
full_forecast[..., 5]
```

也就是 q50，而不是 mean。

原因通常是：

- 中位数对偏态分布更稳健
- 也更适合作为默认 point forecast

### 18.7 为什么自回归时喂回 q50

在 `decode()` 中：

```python
last_renormed_output = renormed_outputs[:, -1, :, self.aridx]
```

其中 `self.aridx = 5`

这表示自回归下一轮输入使用的是：

- 第 5 个通道
- 也就是 q50 / median

直觉上这是在用“最稳健的中心预测”继续滚动，而不是：

- 用 mean
- 或者用某个偏高 / 偏低分位数

### 18.8 两个输出头怎么配合

可以把它们理解成如下分工：

#### 头 1：基础预测头

负责：

- 未来前 128 步的基础预测
- 提供 q50 给自回归滚动
- 在不开 continuous quantile head 时，直接提供 quantile 结果

#### 头 2：长 horizon 分布头

负责：

- 在更长 horizon 上补充和校正 quantile 结构
- 尤其在 `use_continuous_quantile_head=True` 时，提供更平滑、更连续的分位数信息

### 18.9 一句话总结两个输出头

这两个头虽然结构上都只是残差 MLP，但语义完全不同：

- `output_projection_point`：预测“未来值本身”
- `output_projection_quantiles`：预测“围绕 q50 的分布形状 / quantile spread”

因此它们不是“两个等价的 FFN 输出头”，而是：

> 一个负责中心预测和自回归滚动，一个负责更长 horizon 的概率分布建模。

---

## 19. 和这三个主题相关的关键 Python / 张量语法

### 19.1 `dict[str, Sequence[Sequence[float]]]`

例如：

```python
dynamic_numerical_covariates: dict[str, Sequence[Sequence[float]]] | None
```

可以读成：

- 一个字典
- key 是字符串，比如 `"price"`、`"day_of_week"`
- value 是“batch 内每条序列对应的一个序列”

也就是：

```text
{
  "price": [
    [series_1 的 covariate 值],
    [series_2 的 covariate 值],
    ...
  ]
}
```

### 19.2 `collections.defaultdict(list)`

例如：

```python
train_dynamic_numerical_covariates = collections.defaultdict(list)
```

作用是：

- 如果某个 key 第一次出现，不需要先手工初始化 `[]`
- 可以直接 `append`

这很适合把多条序列的 covariate 逐条收集起来。

### 19.3 `[..., None]`

例如：

```python
xreg[..., None]
```

如果：

- `xreg.shape = [H]`

那么：

- `xreg[..., None].shape = [H, 1]`

在这里常用于：

- 把一维修正量 broadcast 到 quantile 维度

例如：

```python
quantile_output + xreg[..., None]
```

表示：

- 每个 horizon 点上的线性修正
- 同时加到所有 quantile 通道上

### 19.4 切片 `a:b`

例如：

```python
covariate_value[(input_len - train_len) : input_len]
covariate_value[input_len:]
```

前者表示：

- 取 context 对应的协变量片段

后者表示：

- 取 future horizon 对应的协变量片段

这是 `xreg` 数据准备的核心切片逻辑。

### 19.5 `zip(..., strict=True)`

在文档前面的辅助脚本里我用过这个写法，这里顺便补一句：

- `zip(a, b, strict=True)` 表示要求两个可迭代对象长度一致
- 如果长度不一致会直接报错

它很适合时间序列代码里防止“预测长度和标签长度默默错位”的 bug。

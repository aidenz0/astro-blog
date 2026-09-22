---
title: LLM 算法 LeetCode（三）：单请求 Decode 与生成策略
author: Aidenz
pubDatetime: 2026-09-22T00:00:00Z
slug: llm-algo-leetcode-task2-decode-sampling-speculative
featured: false
draft: false
series: LLM 算法 LeetCode
seriesOrder: 2
tags:
  - LLM
  - 推理优化
  - KV Cache
  - 采样策略
  - 投机解码
description: 从 KV Cache 的近似账本公式出发，厘清为何它随序列长度线性而非平方增长，再梳理 KV 压力的八类优化手段、Greedy / Temperature / Top-k / Top-p 四种采样策略，以及投机解码中草稿模型与目标模型的分工。
---

## KV Cache 到底缓存了什么

在 Transformer 自回归生成过程中，每生成一个新 token，都要把它加入历史上下文。对于每一层 Attention，历史 token 会产生 Key（K）和 Value（V）。这些 K/V 可以缓存起来，后续生成新 token 时直接复用，不必重新计算历史 token 的 K/V。这就是 **KV Cache**。

它的核心规模近似为：

$$
M_{KV} \approx B \times L \times S \times 2 \times H_{KV} \times D \times Bytes
$$

| 符号 | 含义 |
| --- | --- |
| $B$ | 当前 batch 中的请求 / 序列数 |
| $L$ | Transformer 层数 |
| $S$ | 每条序列当前缓存的 token 数 |
| $2$ | K + V |
| $H_{KV}$ | KV heads 数 |
| $D$ | 每个 head 的维度 |
| $Bytes$ | 每个元素占用字节数，例如 FP16/BF16 = 2 Bytes |

如果是传统 MHA，$H_{KV} = H_Q$；如果是 GQA/MQA，$H_{KV} < H_Q$。一个实用的"账本公式"：

$$
\boxed{ \text{KV显存} \approx \text{层数} \times \text{并发序列数} \times \text{上下文Token数} \times \text{K/V数量} \times \text{KV Head数} \times \text{Head Dim} \times \text{Bytes} }
$$

### 为什么是线性增长，而不是平方

**Attention 计算量和 KV Cache 存储量不是一回事。** 假设当前已经有 $S$ 个历史 token。生成第 $S+1$ 个 token 时，新 token 的 Query 需要和历史所有 K 做 Attention，即 $QK^T$，所以单次 decode 的 Attention 计算量大约随 $S$ 线性增长。

但历史 K/V 已经存起来了。每增加一个 token，只需要增加一个 K(new) 和一个 V(new)，因此 KV Cache 每生成一个 token 只增加固定大小：

$$
\Delta M_{KV} \propto L \times H_{KV} \times D \times 2
$$

累计 $S$ 个 token：$M_{KV} \propto S$，即

$$
\boxed{\text{KV Cache 显存} \propto S}
$$

而不是 $S^2$。那 $S^2$ 从哪里来？它来自 **Attention 矩阵 / Attention 计算**。完整 Attention 是 $\text{Attention}(Q,K,V) = \text{softmax}\left(\dfrac{QK^T}{\sqrt d}\right)V$，当 Q、K 都有 $S$ 个 token 时 $QK^T$ 是 $S \times S$：

$$
\boxed{ \text{Attention 计算 / 中间矩阵} \sim O(S^2) }
$$

但现代推理通常不会把完整 $S \times S$ Attention 矩阵长期保存下来。所以一定要区分：

> **Attention 的计算复杂度可以随序列长度呈平方增长，但 KV Cache 的存储量通常随序列长度线性增长。**

### 并发与 batch 也是线性的

每个请求平均缓存 $S$ 个 token，则 $M_{KV} \propto B \times S$，即 $\boxed{KV\ Cache \sim O(BS)}$。例如 1 个请求 × 8K token 与 8 个请求 × 8K token 相比，KV Cache 理论上大约扩大 8 倍。

### 一个实际例子

假设模型 32 layers、GQA 8 KV heads、head dimension 128、BF16（2 Bytes）、batch 16、每条请求上下文 4096 tokens：

$$
M = 32 \times 16 \times 4096 \times 2 \times 8 \times 128 \times 2 \approx \boxed{8.59\ GB}
$$

这就是为什么部署大模型时——

> **显存不只是被模型权重吃掉，长上下文 + 高并发情况下 KV Cache 可能成为更大的显存压力来源。**

## KV Cache 压力的八类优化

可以按照"到底减少了什么"来分类。

### ① 减少 KV Head 数：MQA / GQA

模型结构层面的优化。传统 MHA 中 $H_{KV} = H_Q$；GQA 中 $H_{KV} < H_Q$；MQA 中 $H_{KV} = 1$，因此 $KV\ Cache \propto H_{KV}$。例如 32 Q heads + 32 KV heads 的 MHA 改为 32 Q heads + 8 KV heads 的 GQA，KV Cache 理论上可缩小到原来的 25%。

- **减少的对象**：KV Cache 本身。
- **新增代价**：需要模型架构支持；若对已有 MHA 模型进行转换，需要模型改造 / 训练或适配；KV 表达能力可能发生变化。

### ② KV Cache 量化

FP16/BF16 → FP8 → INT8，本质是把每个元素的字节数从 2 压到 1，因此 $KV\ Cache$ 显存直接下降约 50%。例如 8 GB 的 BF16 KV Cache 量化到 FP8 约为 4 GB。

- **减少的对象**：每个 KV 元素占用的字节数。
- **新增代价**：量化 / 反量化开销、可能的精度损失、需要硬件和推理框架支持。

### ③ PagedAttention / KV Cache 分页

它的核心不是减少每个 token 的 KV 数量，而是 **减少 KV Cache 的内存碎片和预留浪费**。传统方式可能需要预留连续显存而实际使用率不高；PagedAttention 将 KV Cache 切成 block，让不同请求灵活占用显存块。

- **减少的对象**：内存碎片 + 预分配浪费。
- **新增代价**：block/page 管理、地址映射、调度复杂度增加、可能的间接寻址开销。

### ④ Prefix Cache / Prompt Cache

如果很多请求具有相同前缀（`[系统提示][公司资料][问题A]` / `[系统提示][公司资料][问题B]`），可以缓存公共部分 `[系统提示][公司资料]`。

- **减少的对象**：重复计算和重复存储的 KV。
- **新增代价**：Cache 管理、prefix 匹配、cache eviction、命中率依赖工作负载、prefix 改变后无法复用。

### ⑤ Sliding Window / KV Cache 截断

例如只保留最近 $W = 4096$ 个 token 而非 $S = 32768$ 个，则 $M \propto W$ 而非 $M \propto S$。

- **减少的对象**：历史 token 数量。
- **新增代价**：模型失去较早历史信息。适合长对话、局部上下文任务，不适用于必须完整理解历史信息的场景。

### ⑥ Token 压缩 / Context Compression

例如把 10000 tokens 压缩成 3000 tokens 再送给模型，KV Cache 和 Attention 计算都下降。

- **减少的对象**：实际进入 Transformer 的 token 数量。
- **新增代价**：压缩计算本身有成本、信息损失风险、压缩策略复杂。

### ⑦ Continuous Batching

Continuous Batching **并不直接减少单个请求的 KV Cache 大小**，它主要解决 GPU 利用率、请求调度效率和吞吐率。传统 Static Batch 必须等待最长请求完成，而 Continuous Batching 可以在短请求结束后让新请求立即进入。需要注意的是 $\boxed{Batching\ 通常不会让 KV Cache 从根本上减少}$，甚至并发提高时 KV Cache 总量可能增加。

- **减少的对象**：GPU 空闲 / 调度浪费。
- **新增代价**：调度复杂度。

### ⑧ FlashAttention

FlashAttention 主要优化 Attention 的计算和显存访问，并不是直接降低 KV Cache 的理论账本大小，它优化的是 Attention 中间结果的显存占用、GPU HBM ↔ SRAM 的数据搬运和 Attention Kernel 的计算效率。因此 $\boxed{FlashAttention \neq KV\ Cache 压缩}$。

- **减少的对象**：Attention 中间显存 / IO。
- **新增代价**：Kernel 实现复杂度。

### 方法总结

| 方法 | 主要减少什么 | KV Cache 本身？ | 新增代价 |
| --- | --- | --- | --- |
| GQA/MQA | KV Head 数 | ✅ | 架构 / 模型能力变化 |
| KV Quantization | 每元素字节数 | ✅ | 精度、量化开销 |
| PagedAttention | 碎片 / 预留浪费 | 间接 | 内存管理复杂度 |
| Prefix Cache | 重复 Prefix | ✅ / 重复计算 | Cache 管理 |
| Sliding Window | 历史 token 数 | ✅ | 丢失远期信息 |
| Context Compression | 输入 token 数 | ✅ | 压缩计算、信息损失 |
| Continuous Batching | GPU 空闲 / 调度浪费 | ❌ | 调度复杂度 |
| FlashAttention | Attention 中间显存 / IO | ❌ | Kernel 实现复杂度 |

## Greedy、Temperature、Top-k、Top-p 分别改变了什么

这些操作都发生在 $\boxed{\text{模型输出 Logits → 概率分布 → 选择下一个 Token}}$ 这一阶段。

### ① Greedy

模型输出 A:0.45、B:0.30、C:0.15、D:0.10 时，Greedy 直接选概率最高的 A：

$$
\boxed{ x_{t+1} = \arg\max_x P(x \mid x_{\le t}) }
$$

特点：确定性强、相同输入通常得到相同输出、多样性低。

### ② Temperature

Temperature 修改的是 $\boxed{\text{概率分布的尖锐程度}}$：

$$
P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $z_i$ 是 logits。$T < 1$ 时分布更尖锐（如 A 0.5/B 0.3/C 0.2 → A 0.70/B 0.20/C 0.10）；$T > 1$ 时分布更平坦（→ A 0.40/B 0.32/C 0.28）。因此：

$$
\boxed{ T\uparrow \Rightarrow 概率分布更平坦 \Rightarrow 采样空间扩大 \Rightarrow 多样性\uparrow }
$$

但同时 $\boxed{ T\uparrow \Rightarrow 低概率 Token 被采样的机会\uparrow }$，可能导致输出不稳定、重复 / 跑题、事实错误概率增加、格式遵循能力下降。因此 Temperature 本质上是在调：

> **随机采样的"探索程度"。**

### ③ Top-k

Top-k 不改变模型原始 logits 的形状，而是 **只允许概率最高的 K 个 token 参与采样**。例如在 A 0.35/B 0.25/C 0.18/D 0.10/E 0.07/F 0.05 中设 $k=3$，只保留 A、B、C 后重新归一化。$\boxed{Top\text{-}k 控制候选 Token 数量}$。

### ④ Top-p / Nucleus Sampling

Top-p 不规定固定数量，而是 **从最高概率开始累加，直到累计概率达到 p**。例如 p=0.85 时 A+B+C=0.85，只保留 A、B、C；若模型分布很集中（A 0.85/B 0.08/C 0.04/D 0.03），可能只需要 A；若分布较平，就会保留更多 token。$\boxed{Top\text{-}p 控制"累计概率覆盖范围"}$。

### 四者对比与典型流程

| 方法 | 改变的东西 | 核心作用 |
| --- | --- | --- |
| Greedy | 选择策略 | 永远选最大概率 |
| Temperature | 概率分布形状 | 控制随机性 / 尖锐程度 |
| Top-k | 候选数量 | 只保留前 K 个 |
| Top-p | 候选概率质量 | 保留累计概率达到 p 的集合 |

一个典型采样流程可以理解成：

```
Logits → Temperature → Top-k / Top-p → 重新归一化 → Random Sampling → Next Token
```

## 投机解码（Speculative Decoding）

最核心的一句话：

> **让一个小模型先"猜"多个 token，再让大模型一次性验证这些猜测，从而减少大模型逐 token 解码的次数。**

### 为什么需要它

普通 Decode 中，大模型每生成一个 token 都要单独做一次 Forward，大量计算具有串行性质。如果能一次提交多个候选 token 让大模型并行验证，就有机会减少 Target Model 的串行执行次数，从而降低生成延迟、提高吞吐。

### 基本思路

引入一个小模型 Draft Model，先让它一次猜多个 token（如 $t_1 t_2 t_3 t_4 t_5$），然后交给大模型 Target Model 一次验证：

```
t1 ✓  t2 ✓  t3 ✓  t4 ✓  t5 ✗
```

如果前 4 个 token 被接受，就可以直接复用。草稿模型和目标模型分别承担：

- **Draft Model：负责"猜"**。$\boxed{\text{快速提出候选 Token}}$。模型小、推理快、单 token 成本低。
- **Target Model：负责"验证"**。$\boxed{\text{最终决定哪些 Token 可以接受}}$。它仍然是最终输出质量的决定者。

> **小模型不是替代大模型，而是在帮大模型提前提出候选答案。**

### 为什么能加速

普通 Decode 是大模型串行地一次出一个 token；Speculative Decoding 让小模型一次猜 5 个、大模型一次 Forward 并行验证这 5 个。如果大模型能并行处理这些候选 token，就可以减少 Target Model 的串行执行次数，从而降低延迟、提高吞吐。

### 为什么不是无条件加速

假设 Draft Model 猜 5 个、Target Model 验证得到 ✓✓✓✓✗，接受 4 个，收益较好。但如果第一个就错（✗），后面的 B、C、D、E 基本都失去了意义。因此投机解码的收益与 $\boxed{\text{Draft Model 与 Target Model 的分布一致程度}}$ 高度相关。

### "预测—验收"模型

```
┌──────────────┐
│  Draft Model │  小模型
└──────┬───────┘
       │ 猜多个 Token
       ▼
┌──────────────────┐
│   Target Model   │  大模型
└────────┬─────────┘
         │ 并行验证
   ┌─────┴─────┐
   │           │
 接受         拒绝
   │           │
   ▼           ▼
保留 Draft   从 Target 继续
```

## 一页速记

| 知识点 | 核心结论 |
| --- | --- |
| KV Cache | $M_{KV} \propto B \times S \times L \times H_{KV} \times D \times Bytes$ |
| KV Cache 与序列长度 | 线性增长，不是平方增长 |
| $S^2$ | 主要来自 Attention 计算 / 矩阵规模 |
| GQA/MQA | 减少 KV Head 数 |
| KV Quantization | 减少每个 KV 元素的字节数 |
| PagedAttention | 减少碎片和预留浪费 |
| Prefix Cache | 复用重复 Prefix 的 KV |
| Sliding Window | 限制历史 token 数 |
| Context Compression | 减少输入 token 数 |
| Continuous Batching | 提高 GPU 利用率，不直接减少 KV Cache |
| Greedy | 选择概率最大的 token |
| Temperature | 改变概率分布尖锐程度 |
| Top-k | 保留概率最高的 K 个 token |
| Top-p | 保留累计概率达到 p 的 token 集合 |
| Temperature ↑ | 多样性可能 ↑，稳定性可能 ↓ |
| Speculative Decoding | 小模型猜，大模型验 |
| Draft Model | 快速生成候选 token |
| Target Model | 最终验证并决定输出 |
| 投机解码收益 | 取决于 Draft 与 Target 的一致程度 |

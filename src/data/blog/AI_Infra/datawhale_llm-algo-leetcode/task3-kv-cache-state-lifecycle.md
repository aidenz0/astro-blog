---
title: LLM 算法 LeetCode（四）：KV Cache 状态与生命周期
author: Aidenz
pubDatetime: 2026-09-28T00:00:00Z
slug: llm-algo-leetcode-task3-kv-cache-state-lifecycle
featured: false
draft: false
series: LLM 算法 LeetCode
seriesOrder: 3
tags:
  - LLM
  - 推理优化
  - KV Cache
  - PagedAttention
  - RadixAttention
description: PagedAttention 管"内存怎么分配"（物理层），RadixAttention 管"哪些缓存内容能被跨请求复用"（逻辑层）。两者互补，RadixAttention 实际构建在分页式 KV 分配之上。
---

> **一句话总览**：PagedAttention 管"内存怎么分配"（物理层），RadixAttention 管"哪些缓存内容能被跨请求复用"（逻辑层）。两者互补，RadixAttention 实际构建在分页式 KV 分配之上。

## 一、PagedAttention：为什么分块、解决了什么

### 1.1 连续显存分配的问题

即"为什么不直接给每条序列分配一大段连续内存"：

1. **过度预留（内部浪费）**：生成长度事先未知，通常按 `max_len` 一次性预留整段内存，实际只用一小部分，大量显存被白白占用。
2. **扩容搬迁**：生成中 KV 持续增长，连续区间占满后必须另找更大的连续块，并把整段旧 KV 复制过去（memcpy），成本高且阻塞 GPU 流水线。
3. **外部碎片**：多条并发序列各占一段连续空间，分配/释放交错后产生大量碎片，整体显存利用率很低。
4. **无法共享**：连续分配下，即使多条请求前缀完全相同（并行采样、beam search），也只能各存一份 KV。

### 1.2 分块方案

**PagedAttention**（vLLM，借鉴操作系统虚拟内存分页）：把 KV Cache 切成**固定大小的 block**（vLLM 默认 16 个 token/块），显存按块分配，并用一张 **Block Table（块表，类比页表）** 把逻辑块号映射到物理块号。

### 1.3 逻辑 token 位置、物理 KV block、Block Table 的关系

- **逻辑 token 位置 `pos`** → 逻辑块号 = `pos ÷ block_size`，块内偏移 = `pos mod block_size`；
- **逻辑块号** → 查该序列的 Block Table → 物理块地址；
- **物理地址** = 物理块基址 + 块内偏移 × 每 token 的 (K+V) 字节数；Attention kernel 每次都走这条间接路径读 KV。

示意（`block_size = 4`，以 `pos = 9` 为例）：

```
逻辑 token 序列          Block Table        物理 GPU 显存（可碎片化）
（连续编号）             （每序列一张）

┌────────────────────┐   ┌──────────────┐   ┌──────────────────────────────┐
│ L0: [0 1 2 3]      │──►│ L0 → P3      │──►│ P0(空闲)   P3(已分配) P7(已分配)│
│ L1: [4 5 6 7]      │   │ L1 → P7      │   │ P1(已分配) P2(空闲)   P4(空闲)  │
│ L2: [8 9 10 11] ◄高亮│  │  L2 → P1 ◄高亮│   │ P5(已分配) P9(已分配) P6(空闲)  │
│ L3: [12 13 14 15]  │   │ L3 → P5      │   │ P8(空闲)   P10(空闲)  P11(空闲) │
│ L4: [16 17 18 19]  │   │ L4 → P9      │   └──────────────────────────────┘
│   pos 0–3 / 4–7 …  │   └──────────────┘
└────────────────────┘        ▲
       │                      │ Block Table 把逻辑块号映射到物理块号
       ▼                      │ Attention kernel 每次都走这条间接路径读 KV
 pos=9 → 逻辑块 = 9÷4 = 2（L2），块内偏移 = 9 mod 4 = 1
        → 查 L2 → P1 → 物理地址 = P1 基址 + 1 × 每 token 的 (K+V) 字节数
```

### 1.4 为什么逻辑连续不要求物理连续

因为 Block Table 提供了**间接层**：逻辑地址空间连续编号，物理块可以散落在显存任意位置、甚至与其他请求的块交错。Attention 计算只需要按表寻址，根本不关心物理块在哪里。正确性只取决于"映射正确"，与物理块是否相邻无关。

这层间接换来了四个收益：

1. **消除过度预留**：不再按 `max_len` 预留整段，只按已生成 token 数按块分配，避免内部浪费；
2. **消除外部碎片**：任意空闲块都能用，无需为每条序列找连续大块；
3. **扩容零搬迁**：序列变长只需追加一个新块，无需把整段旧 KV memcpy 到更大连续空间；
4. **块级共享 + 写时复制（COW）**：并行采样 / beam search 共享相同前缀块，引用计数管理，写入时才 COW。

## 二、RadixAttention：前缀树 + 最长前缀匹配

### 2.1 组织方式

**RadixAttention**（SGLang）把 KV Cache 组织成一棵 **radix tree（前缀树 / 字典树）**：

- 每个节点代表一段 token 前缀及其缓存 KV，节点到根的路径就是该前缀；
- 多条请求共享的前缀**全局只存一份**，从根开始分叉；
- 叶子对应完整请求 / 对话。

### 2.2 最长前缀匹配（LPM）在请求处理中的作用

新请求或多轮续问到达时，用请求 token 序列在树上做 **LPM（Longest Prefix Match）**：

1. 找到最长的已缓存前缀（cache hit）；
2. 命中部分**直接复用 KV、跳过 prefill 计算**（prefill 是推理中最重的计算阶段，成本正比于输入 token 数）；
3. 只对未命中后缀做**增量 prefill**；
4. 缓存超限时按策略（LRU / 按 token 加权）淘汰整棵子树回收显存。

示意（4 个请求共享系统提示词与 few-shot 示例）：

```
root（空）
  │ 系统提示词 tokens
  ▼
┌────────────────────┐
│ [A] 系统提示词       │  ← 4 个请求共享，KV 已缓存
└──────────┬─────────┘
  │ few-shot 示例 tokens
  ▼
┌────────────────────┐
│ [B] 系统提示词+示例│  ← 4 个请求共享
└─┬──────────┬───────┘
  │          │          ┌────────────────┐
  │          │          │ "发票怎么开"   │ （未命中）
  │          │          ▼
  │          │        ┌─────────┐
  │          │        │ [F] R3   │ ← R3 只需增量 prefill 这一段
  │          │        └─────────┘
  ▼          ▼
"怎么申请退款"      "订单什么时候发货"
  ▼                  ▼
┌─────────────┐    ┌───────────┐
│ [C] …       │    │ [E] R2    │
└──┬──────┬───┘    └───────────┘
   │      │
   ▼      └─ "人工呢"（续轮新后缀）
  R1（完成）
   │
   └► [D] … → R4（续轮）
       LPM 命中 root→A→B→C，历史绝不重算
```

> 绿色 = LPM 命中、直接复用 KV（跳过 prefill）；蓝色 = 未命中后缀，仅此段做增量 prefill。

### 2.3 典型收益

- **LPM 的作用**：前缀匹配越长，省下的 prefill 计算越多；
- **多轮对话 = 前缀续写**：把上一轮完整路径作为前缀再次匹配，历史绝不重算；
- **缓存淘汰**：显存有上限时按 LRU / 按 token 数加权等策略裁掉节点，整棵子树一起回收。

典型场景（共享 system prompt、few-shot 示例、多轮对话历史）命中率高，显存占用与首 token 延迟同时大幅下降。

## 三、两者对比：前缀复用 vs 内存分配管理

| 维度 | PagedAttention（vLLM） | RadixAttention（SGLang） |
| --- | --- | --- |
| **核心问题** | 显存怎么分配（物理层：分配 / 寻址） | 缓存内容怎么复用（逻辑层：索引 / 共享） |
| **作用层面** | Block Table + 引用计数 + 写时复制 | 前缀树 + 最长前缀匹配 + 淘汰策略 |
| **分配 / 共享单位** | 固定大小 block（如 vLLM 默认 16 token/块），共享以"整块完全相同"为条件 | 任意长度前缀（token 粒度），比块级更细 |
| **主要收益** | 省显存：消除过度预留、外部碎片、扩容搬迁 | 省显存 + 省计算：跳过重复 prefill（成本最高的阶段） |
| **共享时机** | 块级、偶发：beam search / 并行采样等恰好前缀相同的场景 | 系统性：任何跨请求共享的前缀都被树去重并复用 |
| **两者关系** | 互补的两层：PagedAttention 提供"怎么放"的底层（分页分配 + 间接寻址）；RadixAttention 提供"放什么、复用哪段"的决策层；SGLang 实际把 RadixAttention 构建在分页式 KV 分配之上 | 同左 |

**一句话区分**：

- **PagedAttention 回答"内存怎么放"** —— 显存**分配管理**问题（物理层）：分块、按需分配、消除碎片与搬迁、块级共享（COW）。它不关心缓存内容是什么，也不为跨请求复用做索引；
- **RadixAttention 回答"哪些内容值得缓存、如何被共享"** —— **前缀复用**问题（逻辑层）：用前缀树索引缓存，跨任意请求做 token 级前缀去重，通过 LPM 跳过重复计算。

## 附：概念出处

- PagedAttention：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（vLLM，SOSP 2023，arXiv:2309.06180）。
- RadixAttention：*Efficiently Programming Large Language Models using SGLang*（SGLang，arXiv:2312.07104）。

## 一页速记

| 知识点 | 核心结论 |
| --- | --- |
| 连续分配的痛点 | 过度预留、扩容搬迁、外部碎片、无法共享 |
| PagedAttention | KV 切成固定 block + Block Table 逻辑→物理映射 |
| Block Table 作用 | 间接层：逻辑连续不要求物理连续 |
| block_size | vLLM 默认 16 token/块 |
| 块级共享 | 引用计数 + 写时复制（COW） |
| RadixAttention | KV 组织成前缀树，共享前缀全局只存一份 |
| LPM | 最长前缀匹配，命中部分跳过 prefill |
| 多轮对话 | 上一轮路径作为前缀再次匹配，历史不重算 |
| 缓存淘汰 | LRU / 按 token 加权，整棵子树回收 |
| 两者关系 | PagedAttention = 物理层（怎么放）；RadixAttention = 逻辑层（放什么、复用哪段） |
| RadixAttention 落地 | 构建在分页式 KV 分配之上 |

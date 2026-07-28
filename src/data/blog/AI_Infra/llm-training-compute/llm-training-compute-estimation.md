---
title: 大模型训练算力计算指南：从 FLOPs 到 GPU 小时
author: Aidenz
pubDatetime: 2026-07-28T03:25:00Z
slug: llm-training-compute-estimation
featured: true
draft: false
tags:
  - LLM
  - 算力
  - FLOPs
  - Scaling Laws
  - 训练
  - GPU
description: 系统梳理大模型训练所需算力的计算方法，涵盖 FLOPs 估算（6PD 公式）、显存需求（参数 + 优化器 + 激活值）、GPU 利用率与训练时间换算、Chinchilla Scaling Laws，并配以 LLaMA-7B 等实例计算。
---

> 本文整理自 OpenAI Scaling Laws（Kaplan et al., 2020）、DeepMind Chinchilla Scaling Laws（Hoffmann et al., 2022）、EleutherAI Transformer Math 101、Epoch AI 的训练算力估算方法，以及 Kipply 的 Transformer Inference Arithmetic 等资料，力求给出一套自洽且可操作的大模型训练算力计算框架。

## 为什么需要算力计算

大语言模型（LLM）的训练成本动辄数百万美元，而算力估算是以下决策的基础：

- **训练前**：评估需要多少 GPU、训练多久、显存是否足够
- **训练中**：监控实际 MFU（Model FLOPs Utilization），诊断效率瓶颈
- **训练后**：对比不同模型的算力效率，验证 Scaling Laws 预测

算力的核心度量单位是 **FLOP**（浮点运算次数，Floating Point Operations）。衍生单位包括：

| 单位 | 含义 | 适用场景 |
|------|------|---------|
| FLOP | 浮点运算总次数 | 单次前向/后向的计算量 |
| FLOP/s（FLOPS） | 每秒浮点运算次数 | GPU 峰值算力 |
| GPU-hours | GPU 数量 × 小时数 | 工程报价 |
| PetaFLOP-day | $10^{15} \times 86400 \approx 8.64 \times 10^{19}$ FLOP | Scaling Laws 论文常用 |

## 1. 训练总算力：6PD 公式

### 1.1 核心公式

Transformer 语言模型训练所需总算力的经典估算公式为：

$$C \approx 6PD$$

其中：

- $C$：训练总计算量（FLOP）
- $P$：模型参数量（Parameters）
- $D$：训练 token 数（Dataset size）

该公式由 OpenAI Scaling Laws 论文 [Kaplan et al., 2020] 提出并实验验证，DeepMind Chinchilla 论文 [Hoffmann et al., 2022] 进一步确认。

### 1.2 公式推导

6PD 可以拆解为前向传播和反向传播两部分：

$$C = C_{\text{forward}} + C_{\text{backward}}$$

$$C_{\text{forward}} \approx 2PD, \quad C_{\text{backward}} \approx 4PD$$

**为什么是 2PD？** 对每个 token，每个参数参与约 2 次浮点运算（一次乘、一次加，即一个 FMA = 2 FLOP）。因此前向传播处理 $D$ 个 token、$P$ 个参数时，计算量约为 $2PD$。

**为什么反向是前向的 2 倍？** 反向传播需要计算：
1. 损失对每层权重的梯度（约等于一次前向的计算量）
2. 损失对每层输入的梯度，用于向更早的层传播（约等于一次前向的计算量）

因此 $C_{\text{backward}} \approx 2 \times C_{\text{forward}} \approx 4PD$，总计 $C \approx 6PD$。

> **注意**：反向/前向 FLOP 比例在大多数经典架构中接近 2:1。Epoch AI 的实验表明，线性层和卷积层的比例稳定在 2:1，而第一层（无输入梯度传播）为 1:1。使用 Adam 等优化器时，权重更新本身可忽略（大 batch size 下梯度累积后才更新一次）。

### 1.3 实例：LLaMA-7B 训练算力

LLaMA-7B 参数量 $P = 6.74 \times 10^9$，训练数据 1T tokens（$D = 10^{12}$）：

$$C = 6 \times 6.74 \times 10^9 \times 10^{12} = 4.04 \times 10^{22} \text{ FLOP}$$

换算为 PetaFLOP-day：

$$\frac{4.04 \times 10^{22}}{8.64 \times 10^{19}} \approx 468 \text{ PetaFLOP-days}$$

## 2. 从 FLOP 到 GPU 时间

### 2.1 训练时间公式

$$T = \frac{C}{\tau} = \frac{6PD}{N_{\text{GPU}} \times \text{FLOPS}_{\text{GPU}} \times \text{MFU}}$$

其中：

- $T$：训练时间（秒）
- $N_{\text{GPU}}$：GPU 数量
- $\text{FLOPS}_{\text{GPU}}$：单卡峰值算力（查 GPU 规格表）
- $\text{MFU}$：Model FLOPs Utilization，实际计算效率占峰值比例

### 2.2 GPU 峰值算力

常见 GPU 的规格参数（以 NVIDIA 数据手册为准）：

| GPU | 精度 | 峰值算力 | 显存 | 互联 |
|-----|------|---------|------|------|
| V100 | FP16 Tensor Core | 125 TFLOP/s | 32GB | NVLink 300GB/s |
| A100 80GB | BF16 Tensor Core | 312 TFLOP/s | 80GB | NVLink 600GB/s |
| H100 SXM5 | BF16 Tensor Core | 989 TFLOP/s | 80GB | NVLink 900GB/s |
| H20（中国市场） | BF16 Tensor Core | 148 TFLOP/s | 96GB | NVLink 900GB/s |
| H200 | BF16 Tensor Core | 989 TFLOP/s | 141GB | NVLink 900GB/s |
| B200 | BF16 Tensor Core | 2250 TFLOP/s | 192GB | NVLink 1.8TB/s |

> **FMA 约定**：NVIDIA 峰值算力将 Fused Multiply-Add（FMA）计为 2 FLOP。一些 profiler（如 fvcore）将 FMA 计为 1 FLOP，需注意统一口径。
>
> **H20 说明**：H20 是 NVIDIA 为应对美国出口管制而面向中国市场推出的 Hopper 架构阉割版。与 H100 相比，BF16 算力大幅削减（仅为 H100 的约 15%），但显存增至 96GB HBM3，NVLink 互联带宽保持 900GB/s。这使得 H20 在大规模推理场景下仍具优势，但训练效率受限。

### 2.3 MFU（Model FLOPs Utilization）

GPU 峰值算力在实际训练中永远无法完全达到，原因包括：

- **内存带宽瓶颈**：训练通常是 memory-bound 而非 compute-bound
- **多 GPU 通信开销**：分布式训练中梯度同步和激活值通信占用时间
- **处理器-内存速度差距**：内存带宽提升速度远慢于算力提升
- **并行效率**：并非所有计算都能完美并行化

实测 MFU 参考值：

| 设置 | MFU |
|------|-----|
| 单 GPU 简单网络 | 0.30 ~ 0.75 |
| 多 GPU LLM 训练（2021 前） | ~0.33（OpenAI 假设） |
| Megatron-LM on A100 | 0.43 ~ 0.52 |
| GSPMD（Google, LaMDA） | 0.56 ~ 0.62 |
| GPT-NeoX on A100（Flash Attention） | 150~180 TFLOP/s → MFU ≈ 0.48~0.58 |
| 现代 optimized 训练框架 | 0.40 ~ 0.55 |

> **经验法则**：在 A100 上训练 LLM 时，应能达到约 120 TFLOP/s（MFU ≈ 0.38）。若低于 115 TFLOP/s，通常说明模型或硬件配置有问题。

### 2.4 实例：LLaMA-7B 训练时间估算

已知 $C = 4.04 \times 10^{22}$ FLOP，假设使用 8 × A100 80GB：

$$T = \frac{4.04 \times 10^{22}}{8 \times 312 \times 10^{12} \times 0.45} \approx \frac{4.04 \times 10^{22}}{1.12 \times 10^{15}} \approx 3.61 \times 10^7 \text{ 秒}$$

$$\approx 418 \text{ 天} \quad (\text{单机 8 卡})$$

这显然太慢了。实际 LLaMA-7B 使用了 2048 张 A100，训练约 21 天：

$$T = \frac{4.04 \times 10^{22}}{2048 \times 312 \times 10^{12} \times 0.45} \approx 1.4 \times 10^5 \text{ 秒} \approx 1.6 \text{ 天}$$

考虑到 MFU 可能更高以及实际报告的差异，这个量级是合理的。

## 3. 显存需求计算

显存是大模型训练中最关键的约束之一。训练所需总显存为：

$$\text{Total Memory}_{\text{Training}} = \text{Model Memory} + \text{Optimizer Memory} + \text{Gradient Memory} + \text{Activation Memory}$$

### 3.1 模型权重显存

| 精度 | 每参数字节数 | 7B 模型显存 |
|------|------------|-----------|
| INT8 | 1 byte | 7 GB |
| FP16/BF16 | 2 bytes | 14 GB |
| FP32 | 4 bytes | 28 GB |
| 混合精度（FP16 + FP32 副本） | 2 + 4 = 6 bytes | 42 GB |

> **混合精度训练**会同时存储 FP16 权重和 FP32 权重副本（后者计入优化器状态），实际每参数占用 6 bytes。

### 3.2 优化器状态显存

以最常用的 AdamW 优化器为例：

| 优化器 | 每参数字节数 | 组成 |
|--------|------------|------|
| AdamW | 12 bytes | FP32 参数副本(4) + Momentum(4) + Variance(4) |
| 8-bit Adam（bitsandbytes） | 6 bytes | FP32 副本(4) + 8-bit Momentum(1) + 8-bit Variance(1) |
| SGD with momentum | 8 bytes | FP32 副本(4) + Momentum(4) |

### 3.3 梯度显存

| 精度 | 每参数字节数 |
|------|------------|
| FP32 | 4 bytes |
| FP16/BF16 | 2 bytes |

### 3.4 激活值显存

激活值显存与序列长度、batch size、层数成正比。不做激活值重计算时：

$$\text{Memory}_{\text{activations}}^{\text{No Recomputation}} = sbhL \left(10 + \frac{24}{t} + \frac{5as}{ht}\right) \text{ bytes}$$

其中 $s$ 为序列长度，$b$ 为每 GPU 的 batch size，$h$ 为隐藏维度，$L$ 为层数，$a$ 为注意力头数，$t$ 为张量并行度。

| 策略 | 激活值显存 | 额外计算 |
|------|----------|---------|
| 不重计算 | $sbhL(10 + \frac{24}{t} + \frac{5as}{ht})$ | 无 |
| 选择性重计算 | $sbhL(10 + \frac{24}{t})$ | 少量额外前向 |
| 完全重计算 | $2 \cdot sbhL$ | 一次额外前向（$C_{\text{forward}}$ 翻倍） |

### 3.5 总显存估算

以混合精度 + AdamW 优化器为例，不做激活值重计算时：

$$\text{Total} \approx \underbrace{2P}_{\text{模型(FP16)}} + \underbrace{12P}_{\text{优化器}} + \underbrace{2P}_{\text{梯度(FP16)}} + \underbrace{\text{Activations}}_{\text{激活值}} = 16P + \text{Activations}$$

**LLaMA-7B 示例**（$P = 6.74 \times 10^9$）：

$$16 \times 6.74 \text{ GB} \approx 108 \text{ GB} + \text{Activations}$$

单张 A100 80GB 显然放不下。需要借助分布式训练策略。

## 4. 分布式训练的显存分摊

### 4.1 ZeRO 优化阶段

ZeRO（Zero Redundancy Optimizer）通过分片降低冗余：

| 阶段 | 分片对象 | 每卡显存 |
|------|---------|---------|
| ZeRO-0 | 无 | $16P + A$ |
| ZeRO-1 | 优化器状态 | $4P + \frac{12P}{N} + A$ |
| ZeRO-2 | 优化器 + 梯度 | $4P + \frac{14P}{N} + A$ |
| ZeRO-3 | 优化器 + 梯度 + 参数 | $\frac{16P}{N} + A_{\text{live}} + A$ |

其中 $N$ 为 GPU 数（DP degree），$A$ 为激活值。

### 4.2 3D 并行

当结合张量并行（TP）和流水线并行（PP）时：

$$\text{DP Degree} = \frac{N_{\text{GPU}}}{\text{PP Size} \times \text{TP Size}}$$

各部分显存分摊：

$$\text{Total} \approx \frac{\text{Model Memory}}{\text{PP} \times \text{TP}} + \frac{\text{Optimizer Memory}}{N_{\text{GPU}}} + \frac{\text{Activation Memory}}{\text{TP}} + \frac{\text{Gradient Memory}}{\text{PP}}$$

> **注意**：流水线并行不减激活值，且需存储所有在途 micro-batch 的激活值。张量并行与所有 ZeRO 阶段互补。

## 5. Chinchilla Scaling Laws：最优参数与数据配比

### 5.1 计算最优配比

DeepMind 的 Chinchilla 论文通过 400+ 次实验拟合出：在给定算力预算 $C$ 下，最优参数量和数据量为：

$$P^* \approx 0.6 \, C^{0.5}, \quad D^* \approx 10 \, C^{0.5}$$

等价地，**计算最优配比**约为：

$$D^* \approx 20 P^*$$

即每个参数应配约 20 个训练 token。

### 5.2 为什么要超过 Chinchilla 最优

EleutherAI 的实践表明，仅按 $D = 20P$ 训练的模型质量通常较差。原因包括：

- 推理成本考虑：更大的模型推理更贵，但小模型在同等算力下推理更便宜
- 数据质量：高质量数据稀缺时，增加参数比增加数据更有效
- 后训练：RLHF 等后训练阶段需要强基础模型

> **经验法则**：现代 LLM 训练通常不少于 200B tokens，即使 Chinchilla 最优建议更少。实践中倾向于训练能承担的最大模型，配尽可能多的数据。

### 5.3 实际模型的参数-数据配比

| 模型 | 参数量 | 训练 token | $D/P$ | Chinchilla 最优? |
|------|--------|-----------|-------|-----------------|
| Chinchilla | 70B | 1.4T | 20 | ✓（最优） |
| GPT-3 | 175B | 300B | 1.7 | ✗（严重欠训练） |
| LLaMA-7B | 6.7B | 1T | 149 | ✗（过度训练） |
| LLaMA-65B | 65B | 1.4T | 22 | ≈（接近最优） |
| LLaMA-2-7B | 6.7B | 2T | 299 | ✗（极度过度训练） |

LLaMA 系列刻意"过度训练"小模型，使得 7B 模型的性能远超 Chinchilla 预测，推理成本大幅降低。这是一种"推理成本优先"的策略。

## 6. 完整估算流程：以 LLaMA-2-70B 为例

### Step 1: 确认参数和数据

$$P = 68.9 \times 10^9, \quad D = 2 \times 10^{12} \text{ tokens}$$

### Step 2: 计算总算力

$$C = 6PD = 6 \times 68.9 \times 10^9 \times 2 \times 10^{12} = 8.27 \times 10^{23} \text{ FLOP}$$

### Step 3: 估算训练时间

假设 1720 张 H100，MFU ≈ 0.45：

$$T = \frac{8.27 \times 10^{23}}{1720 \times 989 \times 10^{12} \times 0.45} \approx \frac{8.27 \times 10^{23}}{7.65 \times 10^{17}} \approx 1.08 \times 10^6 \text{ 秒} \approx 12.5 \text{ 天}$$

### Step 4: 估算显存

不使用并行时（理论值）：

$$16P = 16 \times 68.9 \approx 1102 \text{ GB}$$

使用 ZeRO-3 + TP=8 + PP=4（$N_{\text{GPU}} = 1720$，DP = 53）：

$$\text{Per-GPU} \approx \frac{1102 \text{ GB}}{32} + \text{Activations}/8 + \text{Overhead} \approx 35 \text{ GB} + \text{Activations}$$

这在 H100 80GB 内是可行的。

## 7. 推理算力估算

训练和推理的算力计算有显著差异。训练是 batch 处理大量 token，而推理是逐 token 自回归生成。

### 7.1 前向传播 FLOPs

每个 token 的前向传播 FLOPs 约为 $2P$（仅前向，无反向）：

$$C_{\text{forward per token}} \approx 2P$$

### 7.2 KV Cache 显存

自回归生成需要缓存历史 token 的 Key 和 Value：

$$\text{KV Cache per token} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \text{ bytes}$$

（因子含义：2 对应 K 和 V，2 对应 FP16 字节数）

### 7.3 Memory-bound vs Compute-bound

推理存在一个关键比值（以 A100 为例）：

$$\frac{\text{FLOPS}}{\text{Memory Bandwidth}} = \frac{312 \times 10^{12}}{1.5 \times 10^{12}} = 208$$

- **batch size < 208**：memory-bound，生成速度受限于权重加载带宽
- **batch size > 208**：compute-bound，生成速度受限于计算能力

这意味着在低 batch size（如单请求）下，GPU 算力大量闲置。

## 8. 实用速查表

### 8.1 快速估算公式

| 目标 | 公式 |
|------|------|
| 训练总算力 | $C = 6PD$ FLOP |
| 训练时间 | $T = \frac{6PD}{N \times \text{FLOPS} \times \text{MFU}}$ |
| 训练显存（混合精度+AdamW） | $\approx 16P + \text{Activations}$ |
| 单 token 前向 | $2P$ FLOP |
| 推理显存（FP16） | $\approx 2P + \text{KV Cache}$ |
| Chinchilla 最优数据量 | $D = 20P$ |

### 8.2 常见模型算力

| 模型 | 参数 | 数据 | 训练算力（PetaFLOP-day） | GPU 估算 |
|------|------|------|----------------------|---------|
| GPT-3 | 175B | 300B | 3638 | 1024 × V100 × 34天 |
| Chinchilla | 70B | 1.4T | 576 | 2048 × A100 × 9天 |
| LLaMA-7B | 6.7B | 1T | 468 | 2048 × A100 × 2天 |
| LLaMA-65B | 65B | 1.4T | 5720 | 2048 × A100 × 21天 |
| LLaMA-2-70B | 70B | 2T | 8440 | 1720 × H100 × 13天 |
| LLaMA-3-70B | 70B | 15T | 63300 | 16384 × H100 × 54天 |

### 8.3 GPU 算力对照

| GPU | BF16 算力 | 显存 | 典型 MFU |
|-----|---------|------|---------|
| V100 | 125 TF | 32GB | 0.25~0.35 |
| A100 | 312 TF | 80GB | 0.40~0.55 |
| H100 | 989 TF | 80GB | 0.45~0.60 |
| H20 | 148 TF | 96GB | 0.45~0.55 |
| B200 | 2250 TF | 192GB | 0.50~0.65（预估） |

## 9. 常见误区与注意事项

### 9.1 FLOP 口径不统一

不同工具对 FMA（Fused Multiply-Add）的计数方式不同：NVIDIA 峰值算力将 FMA 计为 2 FLOP，但部分 profiler 将 FMA 计为 1 FLOP。做估算时要确保口径一致。

### 9.2 峰值算力 ≠ 实际算力

GPU 数据手册列出的是理论峰值，实际训练中受内存带宽、通信开销、kernel 调度等因素影响，MFU 通常只有 0.3~0.6。用峰值算力直接估算会严重低估训练时间。

### 9.3 激活值显存不可忽略

很多人只算"模型 + 优化器 + 梯度"（约 $16P$），忽略激活值。在长序列（如 $s = 4096$）和大 batch 下，激活值可能占总显存的 30% 以上。激活值重计算（Checkpointing）可以用额外计算换显存，但会使前向 FLOPs 翻倍。

### 9.4 Chinchilla 最优 ≠ 实际最优

Chinchilla 给出的是"训练算力效率最优"配比，但实际部署还需考虑推理成本。LLaMA 系列刻意过度训练小模型（$D/P \gg 20$），牺牲训练效率换取推理效率，这在生产环境中往往更经济。

## 参考资料

- **OpenAI Scaling Laws**：Kaplan et al., "Scaling Laws for Neural Language Models", 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **DeepMind Chinchilla**：Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **EleutherAI Transformer Math 101**：Anthony, Biderman, Schoelkopf, 2023. [blog.eleuther.ai/transformer-math](https://blog.eleuther.ai/transformer-math/)
- **Epoch AI - Estimating Training Compute**：Sevilla, Heim, Hobbhahn et al., 2022. [epoch.ai/blog/estimating-training-compute](https://epoch.ai/blog/estimating-training-compute)
- **Epoch AI - Backward-Forward FLOP Ratio**：Hobbhahn, Sevilla, 2021. [epoch.ai/blog/backward-forward-FLOP-ratio](https://epoch.ai/blog/backward-forward-FLOP-ratio)
- **Transformer Inference Arithmetic**：Kipply, 2022. [kipp.ly/blog/transformer-inference-arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
- **Reducing Activation Recomputation**：Korthikanti et al., 2022. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
- **ZeRO**：Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- **Megatron-LM**：Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters", 2021. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

---
title: 大模型算力计算指南：从训练到推理
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
  - 推理
  - GPU
description: 系统梳理大模型训练与推理所需算力的计算方法，涵盖训练 FLOPs 估算（6PD 公式）、训练显存需求、GPU 利用率与训练时间换算、Chinchilla Scaling Laws，以及推理算力（2P/token）、KV Cache 显存、Prefill/Decode 两阶段延迟、memory-bound 与 compute-bound 分析、推理优化技术，并配以 LLaMA-7B/70B 等实例计算。
---

> 本文整理自 OpenAI Scaling Laws（Kaplan et al., 2020）、DeepMind Chinchilla Scaling Laws（Hoffmann et al., 2022）、EleutherAI Transformer Math 101、Epoch AI 的训练算力估算方法、以及 Kipply 的 Transformer Inference Arithmetic 等资料，力求给出一套自洽且可操作的大模型训练与推理算力计算框架。

## 为什么需要算力计算

大语言模型（LLM）的训练成本动辄数百万美元，推理成本在生产环境中同样不可忽视。算力估算是以下决策的基础：

- **训练前**：评估需要多少 GPU、训练多久、显存是否足够
- **训练中**：监控实际 MFU（Model FLOPs Utilization），诊断效率瓶颈
- **训练后**：对比不同模型的算力效率，验证 Scaling Laws 预测
- **推理部署**：评估单卡可服务多少并发请求、延迟和吞吐量如何、需要多少 GPU

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

训练和推理的算力计算有显著差异。训练是 batch 处理大量 token（前向 + 反向），而推理是逐 token 自回归生成（仅前向）。推理的核心挑战不在于总算力，而在于**延迟**和**吞吐量**的权衡。

### 7.1 推理的两阶段：Prefill 与 Decode

LLM 推理分为两个截然不同的阶段：

| 阶段 | 特点 | 瓶颈 | 计算模式 |
|------|------|------|---------|
| **Prefill**（预填充） | 处理输入 prompt 的所有 token，并行计算 | 通常 compute-bound | 类似训练前向，可并行处理整个序列 |
| **Decode**（解码） | 逐 token 自回归生成，每次只处理 1 个 token | 通常 memory-bound | 每步需加载全部权重，计算量极小 |

> **关键区别**：Prefill 阶段一次处理 $s$ 个 token，计算量为 $2Ps$，可充分利用 GPU 算力。Decode 阶段每步只处理 1 个 token，计算量为 $2P$，但需从显存加载全部权重 $2P$ bytes（FP16），因此受限于内存带宽而非算力。

### 7.2 推理 FLOPs：每 token 2P

每个 token 的前向传播 FLOPs 约为 $2P$（仅前向，无反向）：

$$C_{\text{per token}} \approx 2P$$

**推导**：Transformer 的主要计算来自矩阵乘法。每个 token 需要与所有权重矩阵做乘法，而矩阵-向量乘法 $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^n$ 的 FLOPs 为 $2mn$。将所有权重矩阵的 FLOPs 求和，恰好约等于 $2P$。

**逐层分解**（每层每 token）：

| 操作 | FLOPs | 占比 |
|------|-------|------|
| QKV 投影 | $2 \times 3 \times d^2$ | 25% |
| 输出投影 $W_o$ | $2 \times d^2$ | 8.3% |
| MLP $W_1, W_2$ | $2 \times 8 \times d^2$ | 66.7% |
| **合计（每层）** | $2 \times 12 \times d^2$ | 100% |

> 注意力分数计算（$q \cdot k$、softmax、$\text{softmax} \cdot v$）是向量-向量运算，FLOPs 仅为 $O(d)$ 量级，相比矩阵乘法的 $O(d^2)$ 可忽略。

### 7.3 推理显存：权重 + KV Cache

推理所需显存远小于训练，无需优化器状态和梯度：

$$\text{Total Memory}_{\text{Inference}} \approx \text{Model Weights} + \text{KV Cache} + \text{Overhead}$$

#### 模型权重

| 精度 | 每参数字节数 | 7B 模型 | 70B 模型 |
|------|------------|---------|---------|
| INT4 | 0.5 bytes | 3.5 GB | 35 GB |
| INT8 | 1 byte | 7 GB | 70 GB |
| FP16/BF16 | 2 bytes | 14 GB | 140 GB |
| FP32 | 4 bytes | 28 GB | 280 GB |

> **经验公式**（EleutherAI）：推理总显存约为模型权重的 1.2 倍（含约 20% 额外开销）：$\text{Total}_{\text{Inference}} \approx 1.2 \times \text{Model Memory}$。但此公式未考虑 KV Cache，长序列下 KV Cache 可能显著增加显存。

#### KV Cache

自回归生成需要缓存历史 token 的 Key 和 Value 向量，避免重复计算：

$$\text{KV Cache (bytes)} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times s \times b$$

各因子含义：

- **2**：Key 和 Value 两个向量
- **2**：FP16/BF16 每个元素 2 bytes
- $n_{\text{layers}}$：Transformer 层数
- $n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}$：注意力头的总维度
- $s$：序列长度（已生成的 token 数）
- $b$：batch size（并发请求数）

**等价简化公式**：

$$\text{KV Cache} = 4 \times n_{\text{layers}} \times d_{\text{model}} \times s \times b \text{ bytes}$$

**实例：LLaMA-2-70B**（$n_{\text{layers}} = 80$, $d_{\text{model}} = 8192$），FP16：

$$\text{KV Cache per token} = 4 \times 80 \times 8192 = 2,621,440 \text{ bytes} \approx 2.5 \text{ MB}$$

对于 4096 长度序列、batch size 32：

$$\text{KV Cache} = 2.5 \text{ MB} \times 4096 \times 32 \approx 327 \text{ GB}$$

这远超模型权重本身（140 GB），说明**长序列 + 大 batch 下 KV Cache 是显存瓶颈**。

#### 推理总显存实例

| 模型 | 精度 | 权重 | KV Cache（s=2048, b=1） | 总计 |
|------|------|------|------------------------|------|
| LLaMA-7B | FP16 | 14 GB | 1.1 GB | ~15 GB |
| LLaMA-7B | INT4 | 3.5 GB | 1.1 GB | ~5 GB |
| LLaMA-70B | FP16 | 140 GB | 13 GB | ~153 GB |
| LLaMA-70B | INT4 | 35 GB | 13 GB | ~48 GB |

> 单张 A100 80GB 可跑 FP16 的 7B 模型，但 70B 需要 2-4 张或量化为 INT4。

### 7.4 Memory-bound 与 Compute-bound

推理性能的关键在于判断是**内存带宽受限**还是**计算能力受限**。这取决于一个核心比值：

$$R = \frac{\text{GPU FLOPS}}{\text{GPU Memory Bandwidth}}$$

| GPU | 算力 (BF16) | 内存带宽 | 比值 $R$ |
|-----|-----------|---------|---------|
| A100 80GB | 312 TFLOP/s | 1.5 TB/s | 208 |
| H100 SXM5 | 989 TFLOP/s | 3.35 TB/s | 295 |
| H20 | 148 TFLOP/s | 4.0 TB/s | 37 |
| H200 | 989 TFLOP/s | 4.8 TB/s | 206 |
| B200 | 2250 TFLOP/s | 8.0 TB/s | 281 |

**含义**：比值 $R$ 是 memory-bound 和 compute-bound 的分界线。

- **batch size $< R$**：**memory-bound**。每生成一个 token，都需要从显存加载全部权重（$2P$ bytes），但只做 $2P$ FLOPs 的计算。GPU 算力大量闲置，延迟由内存带宽决定。
- **batch size $> R$**：**compute-bound**。多个请求的 KV Cache 可共享权重加载，计算量随 batch 线性增长，GPU 算力成为瓶颈。

> **核心洞察**：在低并发（如单用户请求）下，推理是 memory-bound 的。增加 batch size 可以将多个请求的权重加载"摊薄"，提升算力利用率，但会增加延迟。

### 7.5 延迟计算公式

#### Decode 阶段：单 token 生成延迟

**Memory-bound（小 batch，$b < R$）**：

$$t_{\text{decode}} \approx \frac{2P \times \text{bytes/param}}{N \times \text{Memory Bandwidth}}$$

其中 $N$ 为 GPU 数（张量并行度），权重跨 GPU 分摊。

**Compute-bound（大 batch，$b > R$）**：

$$t_{\text{decode}} \approx \frac{2P \times b}{N \times \text{FLOPS}}$$

#### Prefill 阶段：处理 prompt 延迟

Prefill 一次处理 $s$ 个 token，通常 compute-bound：

$$t_{\text{prefill}} \approx \frac{2P \times s}{N \times \text{FLOPS}}$$

#### 实例：LLaMA-7B on A100

$P = 6.74 \times 10^9$，FP16，单卡 A100（312 TFLOP/s, 1.5 TB/s）：

**Decode（batch=1，memory-bound）**：

$$t_{\text{decode}} = \frac{2 \times 6.74 \times 10^9 \text{ bytes}}{1.5 \times 10^{12} \text{ bytes/s}} \approx 8.99 \text{ ms/token}$$

理论吞吐：$1000 / 8.99 \approx 111 \text{ tokens/s}$

实际受约 10% 的中间操作开销影响，约 22 ms/token，即约 45 tokens/s。

**Decode（batch=256，compute-bound）**：

$$t_{\text{decode}} = \frac{2 \times 6.74 \times 10^9 \times 256}{312 \times 10^{12}} \approx 11.1 \text{ ms/token (per batch)}$$

即每 11.1ms 生成 256 个 token，有效吞吐 $256 / 0.011 \approx 23{,}000 \text{ tokens/s}$。

> 对比可见，从 batch=1 到 batch=256，单 token 延迟从 9ms 增至 11ms（仅增 22%），但吞吐提升了约 200 倍。这就是 batch 推理的价值。

### 7.6 推理吞吐量

**单卡吞吐量**（tokens/s）：

$$\text{Throughput} = \frac{b}{t_{\text{decode}}} \approx \begin{cases} \frac{b \times \text{Bandwidth}}{2P} & \text{if } b < R \\\ \frac{\text{FLOPS}}{2P} & \text{if } b \geq R \end{cases}$$

**关键结论**：

- memory-bound 区域：吞吐量随 batch size 线性增长
- compute-bound 区域：吞吐量达到上限，不再随 batch size 增长
- 最优 batch size 在临界点 $b \approx R$ 附近

### 7.7 推理优化技术

| 技术 | 原理 | 效果 | 适用场景 |
|------|------|------|---------|
| **KV Cache** | 缓存历史 token 的 K/V，避免重计算 | 每步节省 5/6 计算 | 所有自回归推理 |
| **量化（INT8/INT4）** | 降低权重精度 | 显存减半/减至 1/4，带宽需求降低 | 显存受限场景 |
| **PagedAttention（vLLM）** | 分页管理 KV Cache 显存 | 支持 2-4× 更大 batch | 高并发服务 |
| **Flash Attention** | 分块计算注意力 | 减少 HBM 读写，加速长序列 | 长上下文 |
| **Speculative Decoding** | 小模型草拟 + 大模型验证 | 2-3× 解码加速 | 延迟敏感场景 |
| **Continuous Batching** | 动态拼批，请求级粒度 | GPU 利用率从 30%→70%+ | 多请求服务 |
| **Prefix Caching** | 缓存公共前缀的 KV | 减少重复 prefill | 多轮对话/系统提示 |

> **Continuous Batching** 是现代推理引擎（vLLM、TGI）的核心技术。不同于静态 batching 需等所有请求完成才能释放资源，continuous batching 在每个 token 生成步动态插入/移除请求，使 GPU 始终保持高 batch size。

### 7.8 完整推理估算实例：LLaMA-2-70B on 4×A100

**模型参数**：$P = 68.9 \times 10^9$，FP16，$n_{\text{layers}} = 80$，$d_{\text{model}} = 8192$

**Step 1: 显存估算**

权重（4 卡分摊）：$2 \times 68.9 = 137.8 \text{ GB}$，每卡 $34.5 \text{ GB}$

KV Cache（$s = 4096, b = 8$）：$4 \times 80 \times 8192 \times 4096 \times 8 \approx 85.9 \text{ GB}$，每卡 $21.5 \text{ GB}$

总计每卡：$34.5 + 21.5 + \text{Overhead} \approx 60 \text{ GB}$（A100 80GB 可行）

**Step 2: Decode 延迟（batch=8，memory-bound）**

每卡带宽 1.5 TB/s，4 卡总带宽 6.0 TB/s：

$$t_{\text{decode}} = \frac{2 \times 68.9 \times 10^9}{6.0 \times 10^{12}} \approx 23 \text{ ms/token}$$

吞吐：$8 / 0.023 \approx 348 \text{ tokens/s}$

**Step 3: Prefill 延迟（prompt 512 tokens，compute-bound）**

4 卡总算力 $4 \times 312 = 1248$ TFLOP/s：

$$t_{\text{prefill}} = \frac{2 \times 68.9 \times 10^9 \times 512}{1248 \times 10^{12}} \approx 56.5 \text{ ms}$$

**Step 4: 总响应时间**

生成 256 tokens 的请求：$t_{\text{prefill}} + 256 \times t_{\text{decode}} = 0.057 + 256 \times 0.023 \approx 5.9 \text{ s}$

## 8. 实用速查表

### 8.1 快速估算公式

| 目标 | 公式 |
|------|------|
| 训练总算力 | $C = 6PD$ FLOP |
| 训练时间 | $T = \frac{6PD}{N \times \text{FLOPS} \times \text{MFU}}$ |
| 训练显存（混合精度+AdamW） | $\approx 16P + \text{Activations}$ |
| 推理每 token 算力 | $2P$ FLOP |
| 推理显存（FP16） | $\approx 2P + \text{KV Cache}$ |
| KV Cache 显存 | $4 \times n_L \times d \times s \times b$ bytes |
| Decode 延迟（memory-bound） | $\frac{2P}{N \times \text{Bandwidth}}$ |
| Decode 延迟（compute-bound） | $\frac{2P \times b}{N \times \text{FLOPS}}$ |
| Prefill 延迟 | $\frac{2P \times s}{N \times \text{FLOPS}}$ |
| 最优推理 batch | $\approx R = \frac{\text{FLOPS}}{\text{Bandwidth}}$ |
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

### 9.5 推理 ≠ 训练的算力逻辑

训练是 compute-bound（总算力 $6PD$），推理是 memory-bound（单 token 延迟由带宽决定）。两者优化方向完全不同：训练优化 MFU 和通信效率，推理优化 batch size 和显存利用率。用训练的算力思维去估算推理会导致严重误判。

### 9.6 KV Cache 在长序列下可能超过模型权重

很多人估算推理显存时只算模型权重（$2P$），忽略 KV Cache。在长序列（$s = 4096+$）和大 batch 下，KV Cache 可能是模型权重的 2-3 倍。这就是为什么 vLLM 的 PagedAttention 技术如此重要。

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
- **PagedAttention / vLLM**：Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **Flash Attention**：Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Speculative Decoding**：Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

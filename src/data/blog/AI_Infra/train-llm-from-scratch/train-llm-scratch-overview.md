---
title: "从零训练大模型（序）：课程总览"
author: Aidenz
pubDatetime: 2026-07-13T08:18:00Z
slug: train-llm-scratch-overview
featured: true
draft: false
series: 从零训练大模型
seriesOrder: 0
tags:
  - LLM
  - 大模型
  - 从零训练
  - 系列
description: "一门完全零基础的中文系列教程：以 train-llm-from-scratch 仓库为“活教材”，用纯 PyTorch 手写复刻现代大模型的完整训练链路——从张量、Transformer、注意力，一路到预训练、SFT、奖励模型、DPO、PPO、GRPO。本文是 19 章的学习地图与阅读约定。"
---

> 这是一门**完全零基础**的中文系列教程。它以本仓库 `train-llm-from-scratch` 为"活教材",
> 带你从"什么是张量"一路走到能亲手训练出一个会**遵循指令、会推理**的小型大语言模型(LLM)。
> 不需要你预先懂神经网络、不需要你懂 PyTorch —— 这些我们都从头讲起。

## 这门课是给谁的

- 你几乎没有机器学习/深度学习基础,但想真正搞懂"大模型是怎么训练出来的";
- 你会一点点编程概念(知道"变量""函数"是什么)即可,Python 我们会边用边讲;
- 你希望**既理解原理(含完整数学推导),也能亲手把训练跑起来**。

如果你已经懂深度学习,可以直接跳到[第 5 章](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-05-transformer)或本仓库已有的
[英文进阶文档](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/README.md)对应中文版 [`docs/zh/`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/README_zh.md)。

## 你最终会做出什么

这个项目用**纯 PyTorch 手写**(不依赖 `trl`/`peft`/`transformers`)复刻了现代大模型的**完整训练链路**:

```
预训练(Base)  ──►  SFT 指令微调  ──►  奖励模型 ──►  PPO ┐
                       │                              ├─►  GRPO / RLVR(数学推理)
                       └──────────►  DPO / ORPO / KTO ─┘
```

学完这门课,你会理解上面**每一个箭头**在做什么,并且能用项目自带的"小号"配置在自己电脑上(哪怕没有 GPU)把每一步真正跑通。

![后训练全流程](./img/00_overview.png)

## 学习地图(19 章,分 5 个阶段)

建议**严格按顺序**学习,每章都依赖前面的概念。

### 阶段一 · 绝对零基础前置(打地基)

| 章 | 标题 | 你会学到 |
|---|---|---|
| 01 | [环境搭建与第一次运行](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-01-setup) | 装好 Python 环境、把这个项目跑起来、认识它的目录 |
| 02 | [数学与机器学习最小基础](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-02-math-ml-basics) | 向量/矩阵/张量、导数与梯度、概率与对数、"训练"到底是什么 |
| 03 | [PyTorch 极简入门](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-03-pytorch-intro) | 张量运算、自动求导、`nn.Module`、亲手写一个最小训练循环 |

### 阶段二 · 语言模型核心(含完整推导)

| 章 | 标题 | 你会学到 |
|---|---|---|
| 04 | [文本如何变成数字:分词与数据形状](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-04-tokenization) | 分词(tokenization)、token id、上下文窗口、batch 形状 |
| 05 | [解码器 Transformer 骨架](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-05-transformer) | 嵌入、位置编码、残差、LayerNorm、logits、参数量估算 |
| 06 | [注意力机制 · 完整推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-06-attention) | 缩放点积注意力、因果掩码、多头注意力、复杂度 |
| 07 | [训练目标:交叉熵与困惑度 · 完整推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-07-objectives) | 最大似然 → 负对数似然 → 交叉熵、困惑度、标签错位、掩码 |
| 08 | [优化与训练系统](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-08-optimization) | 梯度下降 → Adam → AdamW、学习率调度、梯度累积、混合精度、DDP |
| 09 | [生成与采样](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-09-generation) | 自回归生成、贪心/温度/top-k/top-p 采样 |

### 阶段三 · 预训练实战(亲手跑)

| 章 | 标题 | 你会学到 |
|---|---|---|
| 10 | [数据流水线:从 The Pile 到 HDF5](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-10-data-pipeline) | 下载、预处理、扁平 token 的 HDF5、数据加载窗口 |
| 11 | [预训练你的基座模型 · 动手](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-11-pretraining) | 跑通 `pretrain_base.py`,读懂 loss / 困惑度曲线 |

### 阶段四 · 后训练 / 对齐(每章含完整推导 + 动手)

| 章 | 标题 | 你会学到 |
|---|---|---|
| 12 | [SFT 指令微调 · 含掩码损失推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-12-sft) | 对话模板、只对"回答"部分算损失、序列打包 |
| 13 | [奖励模型 · Bradley-Terry 推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-13-reward-model) | 成对偏好、标量奖励头、Bradley-Terry 损失 |
| 14 | [DPO · 完整目标函数推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-14-dpo) | 直接偏好优化、隐式奖励、为什么能绕开 RL |
| 15 | [PPO · 策略梯度→GAE→裁剪→KL 完整推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-15-ppo) | 策略梯度、actor-critic、GAE、裁剪目标、KL 惩罚 |
| 16 | [GRPO / RLVR · 组相对优势推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-16-grpo) | 可验证奖励、组内相对优势、去掉 critic、课程学习 |

### 阶段五 · 评估、推理与收尾

| 章 | 标题 | 你会学到 |
|---|---|---|
| 17 | [评估(GSM8K)与推理对话 · 动手](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-17-eval-inference) | 用 GSM8K 衡量各阶段、和你的 checkpoint 对话 |
| 18 | [把整条链路跑起来 + 进阶学习路线](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-18-capstone) | 一键全链路、常见报错、接下来学什么 |

## 怎么用这门课(重要约定)

这门课和项目里**已有的中文参考文档**配合使用:

- **本教程(`tutorial/`)** = 给零基础读者的"老师讲解",负责把概念**讲懂**、把推导**展开**、带你**动手**。
- **参考文档(`docs/zh/` 与 `docs/zh/foundations/`)** = 更精炼的"工程速查",每章结尾都会指路到对应参考页,供你深入。

阅读时请注意以下约定:

1. **数学公式**:全部采用标准 LaTeX 美元符号语法——行间公式写成 `$$ ... $$`,行内公式写成 `$ ... $`。
   这种写法被绝大多数编辑器和预览工具(LunarVim 的 `render-markdown.nvim`/`markdown-preview`、VS Code、Typora、Obsidian 等)以及本仓库文档站(MkDocs + MathJax)直接支持。
   如果你在某些 GitHub 旧版页面看到的是 LaTeX 源码而非渲染后的公式,可以在本地用
   `pip install -e ".[docs]"` 后运行 `mkdocs serve`,在浏览器里看到漂亮的排版。
   **每个公式后面都会用大白话解释每个符号的含义**,不会让你对着符号发愁。
2. **代码**:出现的代码片段保持英文原样(变量名、函数名都来自项目真实源码),并标注它在仓库里的**真实路径**,
   例如 `src/models/attention.py`。你随时可以打开对照。
3. **动手命令**:所有命令都能直接复制运行。为了让**没有 GPU** 的同学也能体验,
   动手环节优先使用项目自带的"小号"配置 `configs/smoke/*.json`(模型极小、`device` 设为 `cpu`),
   能在普通笔记本上几秒到几分钟跑完;真正训练出有用的模型则需要 GPU(我们会标注清楚)。
4. **每章结构**:开头给出"本章前置 / 你将学到",结尾给出"小结 + 自测题 + 下一章"。

## 硬件与时间预期

| 目标 | 需要什么 | 大概耗时 |
|---|---|---|
| **理解原理 + 跑通 smoke 小配置** | 一台普通电脑(CPU 即可) | 跟着学几天,跑命令几秒~几分钟 |
| **真正预训练一个 ~400M 基座** | 多张高端 GPU(如 2×H100) | 数天 |
| **完整后训练对齐链路** | 至少一张较大显存 GPU | 数小时~一天 |

> 这门课的价值不在于"刷出多高的分数",而在于**用真实数据、真实代码,把现代大模型从预训练到 RLHF 的每一步亲手走一遍**。

## 术语对照表(全课统一)

| 英文 | 本课译法 | 说明 |
|---|---|---|
| token | 词元 / token | 文本被切成的最小单位;首次出现标注,之后多直接用 token |
| tokenizer | 分词器 | 把文本切成 token 并映射成整数 id 的工具 |
| embedding | 嵌入 / 嵌入向量 | 把离散 id 变成可学习的向量 |
| logits | logits | 模型输出的未归一化得分 |
| softmax | softmax | 把 logits 变成概率分布 |
| attention | 注意力 | Transformer 的核心操作 |
| cross-entropy | 交叉熵 | 预训练/微调的损失函数 |
| gradient descent | 梯度下降 | 用梯度一步步调参数的优化方法 |
| loss | 损失 | 衡量"模型预测有多差"的数值,越小越好 |
| parameter / weight | 参数 / 权重 | 模型里被训练调整的数值 |
| pretraining | 预训练 | 在海量文本上学"语言本身" |
| fine-tuning | 微调 | 在预训练基础上针对具体任务继续训练 |
| SFT | 指令微调 | Supervised Fine-Tuning |
| policy | 策略 | 强化学习里"做决策"的模型(就是我们的 LLM) |
| reward | 奖励 | 强化学习里对一次输出好坏的打分 |
| advantage | 优势 | 某个动作比"平均水平"好多少 |
| checkpoint | 检查点 / checkpoint | 训练过程中保存下来的模型文件 |

## 给你的学习建议

- **不要跳章**。后面每一章都在复用前面的概念(尤其是第 02、05、06、07 章是地基)。
- **边读边动手**。看完一段就去敲一遍命令、改一个数字看看结果变化,理解会牢得多。
- **看不懂公式先别慌**。先读公式后面的大白话解释,理解"它想干嘛",再回头看符号。
- **善用参考文档和源码**。每章末尾的"深入参考"链接,是你巩固和查漏的好去处。

准备好了吗?从 **[第 01 章:环境搭建与第一次运行](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-01-setup)** 开始 👉

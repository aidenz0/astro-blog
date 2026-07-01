---
title: LLM 应用实战精通：系列总览
author: Aidenz
pubDatetime: 2024-01-14T04:00:00Z
slug: applied-llms-mastery-overview
featured: true
draft: false
series: LLM 应用实战精通
seriesOrder: 0
tags:
  - LLM
  - 大模型
  - 系列
description: 一段 10 周的 LLM 实战之旅。本系列系统讲解大语言模型从基础概念、提示工程、微调、RAG，到评估、应用构建、部署、挑战与前沿趋势的全流程，帮你把散落的知识拼成完整图景。
---

## 这是什么系列

这是一个面向<strong>应用落地</strong>的大语言模型（LLM）系列，原素材取自开源课程 *Applied LLMs Mastery 2024*（作者 Aishwarya Naresh Reganti，MIT License），由本站翻译整理为中文。

如果你已经被 LLM 相关的零散概念搞得有点晕——提示工程、微调、RAG、评估、Agent 各说各话，找不到一条从基础到实战的清晰路径——这个系列就是为你准备的。它不追求把数学讲深，而是把"用 LLM 做真实应用"这件事的每个环节串起来，让你既懂原理又知道怎么动手。

## 四大支柱

整个系列围绕四个关键支柱组织：

- **基础**（第 1 周）：LLM 是什么、怎么训练、能做什么、有哪些局限
- **工具与技术**（第 2–5 周）：提示工程、微调、RAG、构建应用所需的工具生态
- **部署与评估**（第 6–9 周）：如何评估、如何端到端构建应用、高级特性与部署、面临的挑战
- **挑战与前沿**（第 9–10 周）：大模型的痛点，以及多模态、Agent、新架构等新兴方向

## 阅读路线

| 顺序 | 标题 | 主题 |
| --- | --- | --- |
| 1 | [LLM 基础与真实世界用例](/posts/ai_infra/applied-llms-mastery/applied-llms-w1p1-foundations) | AI/ML 历史、LLM 规模与训练、典型用例与四类挑战 |
| 2 | [领域与任务自适应方法](/posts/ai_infra/applied-llms-mastery/applied-llms-w1p2-domain-adaptation) | 领域预训练、领域微调与 RAG 的对比取舍 |
| 3 | [提示工程](/posts/ai_infra/applied-llms-mastery/applied-llms-w2-prompting) | 零样本/少样本、CoT/ToT/GoT、ReAct、自洽性等 |
| 4 | [微调大模型](/posts/ai_infra/applied-llms-mastery/applied-llms-w3-finetuning) | 全量微调、PEFT/LoRA、指令微调与 RLHF |
| 5 | [检索增强生成 RAG](/posts/ai_infra/applied-llms-mastery/applied-llms-w4-rag) | 摄取、分块嵌入、向量检索、上下文增强与生成 |
| 6 | [构建 LLM 应用的工具生态](/posts/ai_infra/applied-llms-mastery/applied-llms-w5-tools) | LangChain/LlamaIndex、向量库、模型托管与原型工具 |
| 7 | [LLM 评估技术](/posts/ai_infra/applied-llms-mastery/applied-llms-w6-evaluation) | BLEU/ROUGE/困惑度、MMLU/HELM、LLM-as-a-judge |
| 8 | [动手构建你的 LLM 应用](/posts/ai_infra/applied-llms-mastery/applied-llms-w7-build-app) | 端到端搭建一个 LLM 应用的关键步骤 |
| 9 | [高级特性与部署](/posts/ai_infra/applied-llms-mastery/applied-llms-w8-advanced-deployment) | 函数调用、Agent、缓存、量化与上线权衡 |
| 10 | [大模型面临的挑战](/posts/ai_infra/applied-llms-mastery/applied-llms-w9-challenges) | 幻觉、对抗攻击、对齐、隐私与缓解策略 |
| 11 | [新兴研究趋势](/posts/ai_infra/applied-llms-mastery/applied-llms-w10-research-trends) | 多模态、开源模型、Agent、领域专用模型、MoE/Mamba/RWKV |
| 番外 | [LLM 架构基础](/posts/ai_infra/applied-llms-mastery/applied-llms-w11-architecture) | Seq2Seq、自注意力、编码器-解码器、位置编码 |

> 番外篇（架构基础）数学味稍重，可作为深入研究 LLM 架构时的参考，不影响主线阅读。

## 怎么读

- **想快速建立全局认知**：按顺序读 1→2→5，再跳到 7→9→10，能在一两天内搭起框架。
- **已经在做应用、想补实战**：重点看 3（提示）、5（RAG）、6（工具）、8（构建应用）、9（部署）。
- **关心可靠性与安全**：直奔 7（评估）和 10（挑战）。
- **追前沿**：直接看 11（研究趋势）。

每周内容都包含"5 分钟速览"开头，没时间时只读这部分也能抓住要点；文末的"扩展阅读"和"推荐论文"留给想深挖的读者。

## 致谢与版权

本系列中文版基于 *Applied LLMs Mastery 2024*（原作者 Aishwarya Naresh Reganti，[GitHub 仓库](https://github.com/aishwaryanr/awesome-generative-ai-resources)，MIT License）翻译整理，仅作学习交流用途。原文版权归原作者所有，建议有能力者阅读英文原版。

准备好了的话，从[第一篇：LLM 基础与真实世界用例](/posts/ai_infra/applied-llms-mastery/applied-llms-w1p1-foundations)开始吧。

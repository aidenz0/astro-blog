---
title: Jcode 使用指南：最大化主动的编程 Agent
author: Aidenz
pubDatetime: 2026-07-21T07:45:00Z
slug: jcode-usage-guide
featured: true
draft: false
tags:
  - Jcode
  - Agent
  - AI
  - 工具
description: Jcode 是一个最大化主动 (maximally proactive) 的编程 Agent 与助手，开源在 GitHub。本文系统梳理 Jcode 的核心理念、工具体系、技能（Skills）生态与典型工作流，帮助开发者快速上手并发挥 Agent 的全部潜力。
---

## 什么是 Jcode

[Jcode](https://github.com/1jehuang/jcode) 是一个开源的、**最大化主动**的编码 Agent 与助手。它的设计哲学不是"被动回答问题"，而是：

- 主动理解用户意图，并**采取行动**完成任务；
- 遇到问题优先**修复**而不是仅报告；
- 对长任务具有**持久性 (persistence)**，会一直推进到完成；
- 对不可逆操作（付款、删库、发邮件等）保持审慎。

简言之，Jcode 不是 ChatGPT 式的"问答机器"，而是一个**能动手就不哔哔**的开发伙伴。

## 核心理念

### 1. 主动性与持久性

Jcode 会自主拆解任务、调用工具、验证结果。给定一个目标，它会：

1. 理解意图，列出必要的子任务；
2. 用 `todo` 工具结构化跟踪进度；
3. 调用 `bash`、`edit`、`write` 等工具真正去改代码、跑测试；
4. 验证代码能跑通后再宣称完成。

### 2. 修复优于报告

当 Jcode 发现问题（坏的系统设计、测试失败、构建报错），它会优先**直接动手修**，而不是把锅甩给用户。只有遇到需要决策或不可逆的操作时，才会停下来请求输入。

### 3. 提交即默认 (Commit as you go)

除非用户另有要求，Jcode 默认边做边提交。即使在频繁变动的脏仓库里，也只提交自己的改动，避免污染他人的工作。

## 工具体系一览

Jcode 内置了一套覆盖完整开发链路的工具集，几个关键类别：

| 类别 | 代表工具 | 用途 |
| --- | --- | --- |
| 文件读写 | `read` / `write` / `edit` / `multiedit` | 查看、创建、编辑文件 |
| 代码搜索 | `agentgrep` | grep / find / outline / trace 多模式搜索 |
| Shell | `bash` | 执行命令、脚本、后台任务 |
| 任务管理 | `todo` | 结构化待办与目标评估 |
| 记忆 | `memory` | 跨会话事实、偏好、实体记忆 |
| 浏览器 | `browser` | 自动化网页交互 |
| 调度 | `schedule` | 定时执行未来任务 |
| 多 Agent | `swarm` | 协调多个子 Agent 并行工作 |
| 自我开发 | `selfdev` | 修改 Jcode 自身并重新加载 |
| 会话检索 | `session_search` / `conversation_search` | 搜索历史会话 |

### 后台任务与进度反馈

`bash` 支持 `run_in_background`，长任务可以通过 `JCODE_PROGRESS` / `JCODE_CHECKPOINT` 行报告进度，例如：

```
JCODE_PROGRESS {"percent":42,"message":"Running tests","current":3,"total":10}
```

配合 `bg` 工具的 `wait` / `tail` / `status` 动作，可以可靠地管理长时间运行的构建、测试或部署。

## 技能 (Skills) 生态

Jcode 通过"技能"扩展能力。技能用 `/skillname` 触发，覆盖从创意构思到部署上线的全流程。以下按场景分类列举几类常用技能。

### 创意与规划

- `/brainstorming`：在任何创造性工作（建组件、加功能、改行为）之前，先探索意图、需求和设计。
- `/make-plan`：为多步实现任务创建分阶段的详细计划。
- `/do`：用子 Agent 执行一个分阶段实现计划。
- `/writing-plans`：在有规格/需求时，动代码前先写计划。

### 编码与质量

- `/test-driven-development`：实现任何功能/修任何 bug 前，先写测试。
- `/systematic-debugging`：遇到 bug、测试失败或异常行为时使用，避免直接跳到"修复"。
- `/karpathy-guidelines`：减少常见 LLM 编码错误的行为准则（避免过度复杂化、做手术刀式改动）。
- `/react-best-practices`：TSX 文件编辑后运行的精简质量检查清单。
- `/verification-before-completion`：宣称"完成/修复/通过"前，必须先跑验证命令并确认输出。

### 工作流与协作

- `/using-git-worktrees`：需要在隔离环境做特性开发时使用。
- `/dispatching-parallel-agents`：面对 2+ 个独立任务时并行调度。
- `/subagent-driven-development`：在当前会话中用子 Agent 执行实现计划。
- `/finishing-a-development-branch`：实现完成、测试通过后，决定合并/PR/清理策略。

### 飞书 (Lark) 全家桶

Jcode 提供了一整套飞书集成技能，覆盖 IM、日历、文档、表格、多维表格、幻灯片、妙记、任务、审批、邮箱、视频会议等：

- `/lark-im`：收发消息、管理群聊。
- `/lark-doc` / `/lark-sheets` / `/lark-base` / `/lark-slides`：编辑云文档、表格、多维表格、幻灯片。
- `/lark-calendar` / `/lark-task`：日程与待办。
- `/lark-vc` / `/lark-vc-agent`：视频会议记录与会中事件。
- `/lark-mail`：飞书邮箱起草、发送、回复、搜索。
- `/lark-workflow-standup-report`：生成当日日程 + 未完成任务摘要。

### Vercel / Next.js 生态

如果你在用 Vercel 平台或 Next.js，以下技能非常实用：

- `/nextjs`：App Router 专家指导。
- `/next-forge`：next-forge 单体仓库 SaaS 起手架。
- `/vercel-cli` / `/deployments-cicd` / `/env-vars`：部署、CI/CD、环境变量。
- `/vercel-functions` / `/runtime-cache` / `/vercel-storage`：函数、缓存、存储。
- `/shadcn`：shadcn/ui 组件专家指导。

## 典型工作流示例

### 示例 1：写一个新博客并推送（就是本文）

```text
用户：创建一个 jcode 的使用说明博客，然后 git commit push

Jcode 的步骤：
1. ls + bash 探查项目结构（Astro + AstroPaper 主题）
2. 查看现有博客 frontmatter 格式作为模板
3. 在 src/data/blog/ 下新建 Markdown 文件，遵循 frontmatter schema
4. git add && git commit -m "..." && git push
```

这正是 Jcode"主动性"的体现：用户只给了一句话目标，Agent 自主完成探查、写作、提交、推送全链路。

### 示例 2：修一个 bug

```text
用户：测试挂了，帮我看看

Jcode：
1. /systematic-debugging 先复现、定位
2. /test-driven-development 写一个能复现 bug 的测试
3. 修复实现
4. 跑测试验证（/verification-before-completion）
5. git commit && push
```

### 示例 3：多 Agent 并行

面对多个独立子任务时，用 `swarm` 派发：

```
swarm spawn  → 起一个子 Agent 做子任务 A
swarm spawn  → 起另一个做子任务 B
swarm await_members  → 等所有子 Agent 完成
```

每个子 Agent 会把自己的最终响应报告回给发起者。

## 实用建议

1. **给目标，而不是给步骤。** Jcode 会自己拆解。你只说"加个 RSS 订阅并部署"，它就会去查文档、改代码、跑构建、推送。
2. **用 Skills 显式引导。** 对关键节点（设计、计划、测试、验证）主动 `/brainstorming`、`/make-plan`、`/test-driven-development`、`/verification-before-completion`，质量更稳。
3. **善用记忆。** `memory` 工具可以跨会话记住你的偏好和事实，避免每次重复说明项目背景。
4. **信任但验证。** Jcode 会先验证再宣称完成，但你仍应 review 关键改动，尤其是不可逆操作前。
5. **后台长任务用进度行。** 自己写脚本时加上 `JCODE_PROGRESS {...}` 行，Agent 能更可靠地跟踪和唤醒。

## 结语

Jcode 的价值不在"它知道很多"，而在"它真的去做了"。从探查代码、写测试、修 bug、跑构建，到提交推送、调度子 Agent、操作飞书与 Vercel，它把"想法 → 可运行结果"之间的摩擦降到了最低。

如果你还没试过， clone 下来给它一个真实任务：

```bash
git clone https://github.com/1jehuang/jcode
```

然后告诉它你想做什么，剩下的交给 Agent。

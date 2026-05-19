---
title: Git 常用命令完全指南 - 从入门到进阶
author: Aidenz
pubDatetime: 2025-05-05T04:06:31Z
slug: git-commands-guide
featured: true
draft: false
tags:
  - git
  - 工具
  - 入门教程
description: 系统整理 Git 常用命令，从基础配置、提交与分支，到高级的 HEAD 操作、相对引用和撤销变更，适合初学者和日常查阅。
---

## 引言

Git 是目前最流行的分布式版本控制系统。无论你是个人开发还是团队协作，Git 都能帮你追踪代码变更、管理分支和协作开发。本文系统梳理 Git 从入门到进阶的核心命令。

## 一、初始配置

### 1.1 设置用户名和邮箱

每次 Git 提交都会附带作者信息，所以**第一件事就是配置用户名和邮箱**。

**全局配置（所有仓库生效）：**

```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"
```

**局部配置（仅当前仓库生效）：**

```bash
git config user.name "你的名字"
git config user.email "你的邮箱@example.com"
```

### 1.2 查看配置

```bash
# 查看全局配置
git config --global --list

# 查看当前仓库配置（包含全局 + 局部）
git config --list

# 查看某个具体配置
git config --global user.name
git config --global user.email
```

**全局 vs 局部配置：**

| 配置类型 | 作用范围 | 配置文件位置 | 常用场景 |
|----------|----------|-------------|----------|
| 全局（`--global`） | 所有仓库 | `~/.gitconfig` | 个人用户名、邮箱 |
| 局部（无 flag） | 当前仓库 | `.git/config` | 项目特定配置 |

## 二、仓库基础操作

### 2.1 初始化与克隆

```bash
# 在当前目录初始化 Git 仓库
git init

# 克隆远程仓库
git clone https://github.com/user/repo.git

# 克隆到指定目录
git clone https://github.com/user/repo.git my-folder
```

### 2.2 工作流程概览

Git 的工作区域分为三个部分：

```
工作区（Working Directory）
  │  git add
  ▼
暂存区（Staging Area / Index）
  │  git commit
  ▼
本地仓库（Local Repository）
  │  git push
  ▼
远程仓库（Remote Repository）
```

Git 仓库保存的是文件的**快照**。每次提交时，Git 不会盲目复制整个目录，而是将当前版本与上一个版本对比，只把差异打包存储，因此提交记录非常轻量。

### 2.3 查看状态与差异

```bash
# 查看工作区和暂存区的状态
git status

# 查看工作区与暂存区的差异（未 add 的改动）
git diff

# 查看暂存区与上次提交的差异（已 add 未 commit 的改动）
git diff --staged

# 查看提交历史
git log

# 简洁的一行格式
git log --oneline

# 图形化显示分支
git log --oneline --graph --all
```

## 三、提交（commit）

### 3.1 基本提交

```bash
# 1. 将文件添加到暂存区
git add <file>          # 添加单个文件
git add .               # 添加所有改动

# 2. 提交到本地仓库
git commit -m "提交说明"

# 添加所有已跟踪文件的改动并提交（跳过 add）
git commit -am "提交说明"
```

### 3.2 提交规范

好的提交信息能让团队协作更顺畅：

```bash
# 推荐格式：<类型>: <简短描述>
git commit -m "feat: 添加用户登录功能"
git commit -m "fix: 修复首页加载缓慢问题"
git commit -m "docs: 更新 README 文档"
git commit -m "refactor: 重构用户模块代码"
```

**常用提交类型：**

| 类型 | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档变更 |
| `style` | 代码格式（不影响逻辑） |
| `refactor` | 重构（既不修 bug 也不加功能） |
| `test` | 测试相关 |
| `chore` | 构建/工具变更 |

## 四、分支（branch）

### 4.1 分支的概念

Git 的分支非常轻量——它只是一个指向某个提交记录的指针。因此：

> 早建分支！多用分支！

创建再多的分支也不会造成存储或内存上的开销，按逻辑分解工作到不同的分支比维护臃肿的分支简单得多。

### 4.2 分支操作

```bash
# 查看所有分支（* 号表示当前分支）
git branch

# 创建新分支
git branch <branch-name>

# 切换分支
git switch <branch-name>         # Git 2.23+ 推荐
git checkout <branch-name>       # 旧写法

# 创建并切换到新分支（一步到位）
git switch -c <branch-name>      # Git 2.23+ 推荐
git checkout -b <branch-name>    # 旧写法

# 删除分支（已合并的分支）
git branch -d <branch-name>

# 强制删除分支（未合并也删除）
git branch -D <branch-name>

# 重命名分支
git branch -m <old-name> <new-name>
```

### 4.3 合并分支（merge）

将两个分支合并到一起，会产生一个有两个 parent 节点的特殊提交记录。

```bash
# 1. 切换到目标分支（你想把改动合并到的分支）
git checkout main

# 2. 将 feature 分支合并到 main
git merge feature
```

**合并流程图：**

```
合并前：
main:    A ← B ← C
                ↖
feature:         D ← E

合并后：
main:    A ← B ← C ← F (merge commit)
                ↖     ↗
feature:         D ← E
```

### 4.4 变基（rebase）

Rebase 是另一种合并方式，它将一系列提交"复制"到目标分支的顶部，创造更线性的提交历史。

```bash
# 将当前分支变基到 main 上
git checkout feature
git rebase main
```

**merge vs rebase：**

| 特性 | merge | rebase |
|------|-------|--------|
| 提交历史 | 保留分支结构，有合并提交 | 线性历史，更清晰 |
| 是否改写历史 | 否 | 是（复制提交） |
| 适用场景 | 公共分支、保留完整历史 | 个人分支、整理历史 |
| 冲突处理 | 一次解决 | 逐个提交解决 |

> **注意：** 不要对已经推送到远程的公共分支使用 rebase，因为 rebase 会改写提交历史。

## 五、远程操作

### 5.1 远程仓库管理

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add origin https://github.com/user/repo.git

# 修改远程仓库地址
git remote set-url origin https://github.com/user/new-repo.git
```

### 5.2 推送与拉取

```bash
# 推送到远程（-u 设置上游分支，之后只需 git push）
git push -u origin main
git push

# 从远程拉取（自动合并）
git pull

# 拉取但不合并（只更新本地跟踪信息）
git fetch

# 拉取远程分支并创建本地分支
git switch -c feature origin/feature
```

**fetch vs pull：**

| 命令 | 作用 | 是否自动合并 |
|------|------|-------------|
| `git fetch` | 下载远程更新到本地跟踪分支 | 否 |
| `git pull` | 下载 + 合并到当前分支 | 是（等同于 fetch + merge） |

## 六、HEAD 与在提交树上移动

### 6.1 什么是 HEAD？

HEAD 是一个指向**当前所在分支**的引用，也就是你正在其基础上工作的提交记录。

```
HEAD → main → C3（最新提交）
```

大多数修改提交树的 Git 命令都是从改变 HEAD 的指向开始的。

### 6.2 移动 HEAD

```bash
# 直接移动 HEAD 到指定提交（通过哈希值）
git checkout <commit-hash>

# 哈希值只需提供能唯一标识的前几个字符
git checkout fed2    # 而不是 fed2a3b4c5d6...
```

## 七、相对引用

通过哈希值移动不方便，Git 引入了**相对引用**，从一个已知位置（如分支名或 HEAD）开始计算。

### 7.1 基本语法

```bash
# ^ 表示向上移动 1 个提交
git checkout main^        # main 的父提交
git checkout main^^       # main 的祖父提交

# ~<num> 表示向上移动 num 个提交
git checkout main~3       # main 的第 3 代祖先
```

**相对引用示例：**

```
提交历史：C1 ← C2 ← C3 ← C4 (main, HEAD)

main^   → C3
main~2  → C2
main~3  → C1
HEAD~1  → C3（等同于 main^）
```

### 7.2 强制移动分支

```bash
# 将分支强制指向某个提交
git branch -f main HEAD~3    # 将 main 回退 3 个提交
```

> `git branch -f` 会**强制移动分支指针**，不管当前分支历史是否一致。谨慎使用。

## 八、撤销变更

### 8.1 git reset（本地撤销）

`git reset` 通过回退分支指针来撤销提交，相当于"改写历史"。

```bash
# 回退 1 个提交（保留改动在工作区）
git reset HEAD~1

# 回退 1 个提交（保留改动在暂存区）
git reset --soft HEAD~1

# 回退 1 个提交（彻底丢弃改动）
git reset --hard HEAD~1
```

**reset 的三种模式：**

| 模式 | HEAD | 暂存区 | 工作区 | 用途 |
|------|------|--------|--------|------|
| `--soft` | 移动 | 不变 | 不变 | 撤销提交，保留所有改动 |
| `--mixed`（默认） | 移动 | 重置 | 不变 | 撤销提交和 add |
| `--hard` | 移动 | 重置 | 重置 | 彻底回退到某个状态 |

> **注意：** `git reset` 只适用于本地分支。已经推送到远程的提交不要用 reset，应该用 revert。

### 8.2 git revert（安全撤销）

`git revert` 创建一个**新的提交**来撤销指定提交的改动，不会改写历史。

```bash
# 撤销最近一次提交
git revert HEAD

# 撤销指定提交
git revert <commit-hash>
```

**reset vs revert：**

| 特性 | reset | revert |
|------|-------|--------|
| 是否改写历史 | 是 | 否 |
| 是否创建新提交 | 否 | 是 |
| 适用场景 | 本地未推送的提交 | 已推送到远程的提交 |
| 安全性 | 需谨慎 | 安全 |

```
使用 revert 后：

原始：C1 ← C2 ← C3 (main)
revert C2 后：C1 ← C2 ← C3 ← C2' (main)
                              ↑
                    C2' 的状态与 C1 相同
```

## 九、常用命令速查表

### 日常开发

| 场景 | 命令 |
|------|------|
| 查看状态 | `git status` |
| 添加到暂存区 | `git add .` |
| 提交 | `git commit -m "说明"` |
| 推送 | `git push` |
| 拉取最新代码 | `git pull` |
| 查看历史 | `git log --oneline` |

### 分支操作

| 场景 | 命令 |
|------|------|
| 查看分支 | `git branch` |
| 创建分支 | `git branch <name>` |
| 切换分支 | `git switch <name>` |
| 创建并切换 | `git switch -c <name>` |
| 合并分支 | `git merge <name>` |
| 删除分支 | `git branch -d <name>` |

### 撤销操作

| 场景 | 命令 |
|------|------|
| 撤销工作区改动 | `git checkout -- <file>` |
| 取消暂存 | `git restore --staged <file>` |
| 撤销上次提交（保留改动） | `git reset --soft HEAD~1` |
| 彻底回退 | `git reset --hard HEAD~1` |
| 安全撤销已推送的提交 | `git revert HEAD` |

## 学习检查清单

- [ ] 能配置 Git 用户名和邮箱
- [ ] 理解工作区、暂存区、仓库的关系
- [ ] 掌握 add、commit、push 的基本流程
- [ ] 能创建、切换、合并分支
- [ ] 理解 merge 和 rebase 的区别
- [ ] 掌握 HEAD 和相对引用的用法
- [ ] 理解 reset 和 revert 的区别与适用场景

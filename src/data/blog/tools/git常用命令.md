---
title: git常用命令
author: Aidenz
pubDatetime: 2024-06-29T04:06:31Z
slug:
featured: true
draft: false
tags:
  - git
description: git常用命令
---
# git 设置用户名和邮箱

全局配置用户名和邮箱

  

全局配置适用于所有仓库：

```shell

git config --global user.name "你的用户名"

git config --global user.email "你的邮箱@example.com"

```

**验证配置是否成功：**

```shell

git config --global user.name

git config --global user.email

```

  局部设置（仅适用于当前仓库）
```Bash
git config user.name "你的名字"
git config user.email "你的邮箱@example.com"
```
**查看当前配置**：
```Bash
# 查看全局配置
git config --global --list

# 查看当前仓库配置
git config --list
```
# 基础

Git 仓库中的提交记录保存的是你的目录下所有文件的快照，就像是把整个目录复制，然后再粘贴一样，但比复制粘贴优雅许多！

  

Git 希望提交记录尽可能地轻量，因此在你每次进行提交时，它并不会盲目地复制整个目录。条件允许的情况下，它会将当前版本与仓库中的上一个版本进行对比，并把所有的差异打包到一起作为一个提交记录。

## git commit

将暂存区的内容添加到本地仓库中

```shell

git commit

```

## git branch

Git 的分支也非常轻量。它们只是简单地指向某个提交记录 —— 仅此而已。所以许多 Git 爱好者传颂：

  

```

早建分支！多用分支！

```

  

这是因为即使创建再多的分支也不会造成储存或内存上的开销，并且按逻辑分解工作到不同的分支要比维护那些特别臃肿的分支简单多了。

  

在将分支和提交记录结合起来后，我们会看到两者如何协作。现在只要记住使用分支其实就相当于在说：“我想基于这个提交以及它所有的 parent 提交进行新的工作。”

创建分支

```shell

git branch <name>

```

  
  

下面的命令会让我们在提交修改之前先切换到新的分支上

  

```

git checkout <name>

```

>在 Git 2.23 版本中，引入了一个名为 `git switch` 的新命令，最终会取代 `git checkout`，因为 `checkout` 作为单个命令有点超载（它承载了很多独立的功能）。

  

有个更简洁的方式：如果你想创建一个新的分支同时切换到新创建的分支的话，可以通过 `git checkout -b <your-branch-name>` 来实现。

## git merge

接下来咱们看看如何将两个分支合并到一起。就是说我们新建一个分支，在其上开发某个新功能，开发完成后再合并回主线。

  

咱们先来看一下第一种方法 —— `git merge`。在 Git 中合并两个分支时会产生一个特殊的提交记录，它有两个 parent 节点。翻译成自然语言相当于：“我要把这两个 parent 节点本身及它们所有的祖先都包含进来。”

将bugFix 分支 合并到main中

```shell

git checkout bugFix

git merge main

```

## git rebase

第二种合并分支的方法是 `git rebase`。Rebase 实际上就是取出一系列的提交记录，“复制”它们，然后在另外一个地方逐个的放下去。

  

Rebase 的优势就是可以创造更线性的提交历史，这听上去有些难以理解。如果只允许使用 Rebase 的话，代码库的提交历史将会变得异常清晰。

将bugFix分支合并到main：

```shell

git checkout bugFix

git rebase main

```

  

# 高级

## 在树上移动

我们首先看一下 “HEAD”。 HEAD 是一个对当前所在分支的符号引用 —— 也就是指向你正在其基础上进行工作的提交记录。

  

HEAD 总是指向当前分支上最近一次提交记录。大多数修改提交树的 Git 命令都是从改变 HEAD 的指向开始的。

  

HEAD 通常情况下是指向分支名的（如 bugFix）。在你提交时，改变了 bugFix 的状态，这一变化通过 HEAD 变得可见。

  

```shell

`git checkout <commit-hash>`

```

## 相对引用

通过指定提交记录哈希值的方式在 Git 中移动不太方便。在实际应用时，并没有漂亮的可视化提交树供你参考，所以你就不得不用 `git log` 来查查看提交记录的哈希值。

  

并且哈希值在真实的 Git 世界中也会更长。

比较令人欣慰的是，Git 对哈希的处理很智能。你只需要提供能够唯一标识提交记录的前几个字符即可。因此我可以仅输入`fed2` 而不是上面的一长串字符。

  

正如我前面所说，通过哈希值指定提交记录很不方便，所以 Git 引入了相对引用。这个就很厉害了!

  

使用相对引用的话，你就可以从一个易于记忆的地方（比如 `bugFix` 分支或 `HEAD`）开始计算。

  

相对引用非常给力，这里我介绍两个简单的用法：

  

- 使用 `^` 向上移动 1 个提交记录

- 使用 `~<num>` 向上移动多个提交记录，如 `~3`

HEAD在main分支向上移动一个提交记录：

```bash

git checkout main

git checkout main^

```

HEAD在main分支上向上移动`<num>`个提交记录

```bash

git checkout main~<num>

```

> `git branch -f` 的含义是 **强制移动（重置）分支指针**，让某个分支直接指向指定的提交（commit），**不会管当前分支历史是否一致**。

> ```bash

> git branch -f <分支名> <提交ID | 另一个分支>

> ```

  
  

## 撤销变更

在 Git 里撤销变更的方法很多。和提交一样，撤销变更由底层部分（暂存区的独立文件或者片段）和上层部分（变更到底是通过哪种方式被撤销的）组成。我们这个应用主要关注的是后者。

  

主要有两种方法用来撤销变更 —— 一是 `git reset`，还有就是 `git revert`。接下来咱们逐个进行讲解。

### git reset

`git reset` 通过把分支记录回退几个提交记录来实现撤销改动。你可以将这想象成“改写历史”。`git reset` 向上移动分支，原来指向的提交记录就跟从来没有提交过一样。

本地分支中使用 `git reset` 很方便，但是这种“改写历史”的方法对大家一起使用的远程分支是无效的。

```bash

git reset HEAD~1

```

### git revert

```bash

git revert HEAD

```

在我们要撤销的提交记录后面居然多了一个新提交！这是因为新提交记录 `C2'` 引入了**更改** —— 这些更改刚好是用来撤销 `C2` 这个提交的。也就是说 `C2'` 的状态与 `C1` 是相同的。

  

revert 之后就可以把你的更改推送到远程仓库与别人分享
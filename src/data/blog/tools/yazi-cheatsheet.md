---
title: Yazi 终端文件管理器快捷键速查表
author: Aidenz
pubDatetime: 2026-06-30T04:06:31Z
slug: yazi-cheatsheet
featured: false
draft: false
tags:
  - Yazi
  - 工具
  - 终端
  - 速查表
description: Yazi 是基于 Rust 的高性能终端文件管理器，快捷键大量沿用 Vim 逻辑。本文用一屏速查表 + 分类详解的形式，整理覆盖 90% 日常场景的核心指令。
---

[Yazi](https://github.com/sxyazi/yazi) 是用 Rust 编写的高性能终端文件管理器，快捷键大量沿用 Vim/Neovim 逻辑。下面的核心速查表覆盖 90% 日常场景，**一屏即可查全，无需下滑**。

<div class="yazi-cheat not-prose">
  <section class="yz-card">
    <h3>✅ 基本导航</h3>
    <dl>
      <dt><kbd>j</kbd> <kbd>↓</kbd></dt><dd>向下移动</dd>
      <dt><kbd>k</kbd> <kbd>↑</kbd></dt><dd>向上移动</dd>
      <dt><kbd>l</kbd> <kbd>→</kbd></dt><dd>进入目录 / 打开</dd>
      <dt><kbd>h</kbd> <kbd>←</kbd></dt><dd>返回上级目录</dd>
      <dt><kbd>gg</kbd></dt><dd>跳到顶部</dd>
      <dt><kbd>G</kbd></dt><dd>跳到底部</dd>
      <dt><kbd>K</kbd> <kbd>J</kbd></dt><dd>上 / 下翻半页</dd>
    </dl>
  </section>

  <section class="yz-card">
    <h3>📂 选择 &amp; 打开</h3>
    <dl>
      <dt><kbd>Space</kbd></dt><dd>选中 / 取消</dd>
      <dt><kbd>v</kbd></dt><dd>可视连续选择</dd>
      <dt><kbd>V</kbd></dt><dd>退出选择</dd>
      <dt><kbd>Ctrl</kbd>+<kbd>a</kbd></dt><dd>全选</dd>
      <dt><kbd>Ctrl</kbd>+<kbd>r</kbd></dt><dd>反选</dd>
      <dt><kbd>Enter</kbd> <kbd>o</kbd></dt><dd>打开 / 进入</dd>
      <dt><kbd>O</kbd></dt><dd>选择打开方式</dd>
    </dl>
  </section>

  <section class="yz-card">
    <h3>✂️ 文件操作</h3>
    <dl>
      <dt><kbd>y</kbd></dt><dd>复制（yank）</dd>
      <dt><kbd>x</kbd></dt><dd>剪切</dd>
      <dt><kbd>p</kbd></dt><dd>粘贴</dd>
      <dt><kbd>P</kbd></dt><dd>粘贴并覆盖</dd>
      <dt><kbd>Y</kbd> <kbd>X</kbd></dt><dd>取消复制 / 剪切</dd>
      <dt><kbd>d</kbd></dt><dd>删到回收站</dd>
      <dt><kbd>D</kbd></dt><dd>永久删除</dd>
      <dt><kbd>r</kbd></dt><dd>重命名</dd>
      <dt><kbd>a</kbd></dt><dd>新建（<kbd>/</kbd> 结尾建目录）</dd>
      <dt><kbd>.</kbd></dt><dd>显示 / 隐藏隐藏文件</dd>
    </dl>
  </section>

  <section class="yz-card">
    <h3>🧰 高级 &amp; 标签页</h3>
    <dl>
      <dt><kbd>:</kbd> <kbd>;</kbd></dt><dd>执行 shell 命令</dd>
      <dt><kbd>f</kbd></dt><dd>过滤 / 搜索</dd>
      <dt><kbd>z</kbd></dt><dd>zoxide 跳转</dd>
      <dt><kbd>Z</kbd></dt><dd>fzf 模糊查找</dd>
      <dt><kbd>t</kbd></dt><dd>新建标签页</dd>
      <dt><kbd>1</kbd>–<kbd>9</kbd></dt><dd>切换标签页</dd>
      <dt><kbd>[</kbd> <kbd>]</kbd></dt><dd>上 / 下标签页</dd>
    </dl>
  </section>

  <section class="yz-card">
    <h3>📋 复制路径（<kbd>c</kbd> 后接）</h3>
    <dl>
      <dt><kbd>cc</kbd></dt><dd>绝对路径</dd>
      <dt><kbd>cd</kbd></dt><dd>所在目录路径</dd>
      <dt><kbd>cf</kbd></dt><dd>文件名</dd>
      <dt><kbd>cn</kbd></dt><dd>文件名（无扩展名）</dd>
    </dl>
  </section>
</div>

<style>
  .yazi-cheat {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
    gap: 0.75rem;
    margin: 1.5rem 0;
  }
  .yazi-cheat .yz-card {
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.6rem 0.75rem;
    background: color-mix(in srgb, var(--muted) 18%, transparent);
  }
  .yazi-cheat .yz-card h3 {
    margin: 0 0 0.4rem;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--accent);
  }
  .yazi-cheat dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.2rem 0.6rem;
    margin: 0;
    font-size: 0.8rem;
    line-height: 1.4;
  }
  .yazi-cheat dt {
    margin: 0;
    white-space: nowrap;
    text-align: right;
  }
  .yazi-cheat dd {
    margin: 0;
    color: color-mix(in srgb, var(--foreground) 85%, transparent);
  }
  .yazi-cheat kbd {
    display: inline-block;
    padding: 0.05rem 0.35rem;
    font-family: var(--font-google-sans-code), monospace;
    font-size: 0.72rem;
    line-height: 1.4;
    border: 1px solid var(--border);
    border-radius: 0.25rem;
    background: var(--background);
    box-shadow: 0 1px 0 var(--border);
  }
</style>

> 💡 上表覆盖日常高频操作。下方是分类详解，需要更完整的说明时再下滑查看。

## Table of contents

## 📖 详细功能说明

### 1. 基本导航

Yazi 的移动完全遵循 Vim 的 `hjkl`：`j`/`k` 上下移动光标，`l` 进入选中的目录或打开文件，`h` 返回上级目录，同时也支持方向键。

- `gg` 跳到列表第一项，`G` 跳到最后一项；
- `J` / `K` 向下 / 向上翻半页，适合在长列表里快速定位。

### 2. 选择 &amp; 打开

- `Space` 切换单个文件的选中状态，可逐个挑选；
- `v` 进入可视模式后移动光标即可连续选择，`V` 取消选择；
- `Ctrl+a` 全选、`Ctrl+r` 反选，配合批量操作很高效；
- `Enter` 或 `o` 用默认程序打开，`O` 弹出菜单让你选择打开方式。

### 3. 文件操作

复制 / 移动遵循"先标记、后粘贴"的两步流程：

- 先用 `y`（复制）或 `x`（剪切）标记文件，再到目标目录按 `p` 粘贴；
- `P` 在粘贴时覆盖同名文件；`Y` / `X` 可取消之前的复制 / 剪切标记；
- `d` 删除到回收站（可恢复），`D` 永久删除（不可恢复，谨慎使用）；
- `r` 重命名；`a` 新建，名称结尾带 `/` 会创建目录而非文件；
- `.` 切换隐藏文件的显示。

### 4. 高级功能 &amp; 标签页

- `:` 或 `;` 在当前目录执行任意 shell 命令；
- `f` 过滤当前目录，`z` 借助 [zoxide](https://github.com/ajeetdsouza/zoxide) 智能跳转，`Z` 用 fzf 模糊查找；
- `t` 新建标签页，数字键 `1`–`9` 直接切换，`[` / `]` 在标签页间前后切换。

#### 复制路径（按 `c` 后接以下按键）

在处理脚本或粘贴路径时非常实用：

- `cc` 复制文件的绝对路径；
- `cd` 复制文件所在目录的路径；
- `cf` 复制文件名（含扩展名）；
- `cn` 复制不含扩展名的文件名。

> 💡 提示：Yazi 的快捷键逻辑大体沿用 Vim/Neovim，熟悉 Vim 的用户几乎零成本上手。更多高级配置（自定义键位、插件、主题）可参考 [Yazi 官方文档](https://yazi-rs.github.io/) 与 [GitHub 仓库](https://github.com/sxyazi/yazi)。不同版本默认键位可能略有差异，以 `~/.config/yazi/keymap.toml` 实际配置为准。

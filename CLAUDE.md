# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 AstroPaper 模板的个人技术博客，使用 Astro 6 + TailwindCSS 4 构建。博客内容存储在 `src/data/blog/` 目录下，支持中文文章、数学公式（KaTeX）、代码高亮和全文搜索。站点地址：https://aidenz.eu.org/

## 命令

### 项目脚本（pnpm）

```bash
# 开发
pnpm install          # 安装依赖（首次或拉取后依赖有变动时）
pnpm run dev          # 启动开发服务器 (localhost:4321)，支持热更新

# 构建（类型检查 → 构建 → 生成 Pagefind 搜索索引 → 复制到 public/）
pnpm run build
pnpm run preview      # 本地预览构建产物（验证线上效果）

# 代码质量
pnpm run format:check # Prettier 格式检查
pnpm run format       # Prettier 格式化
pnpm run lint         # ESLint 检查
pnpm run sync         # 生成 Astro 模块类型定义（修改 content.config.ts 后运行）
```

### Astro CLI 常用指令

通过 `pnpm astro <command>` 调用（或 `pnpm run astro <command>`）：

```bash
pnpm astro dev              # = pnpm run dev
pnpm astro build            # 仅构建（不含 Pagefind 索引，索引用 pnpm run build）
pnpm astro preview          # 预览 dist/
pnpm astro check            # 类型检查（构建前会自动跑；缺必填 frontmatter 会报错）
pnpm astro sync             # 重新生成 astro:content 等类型定义
pnpm astro add <integration># 添加官方集成（如 pnpm astro add react）
pnpm astro info             # 输出环境信息（提 issue 时附上）
pnpm astro --help           # 查看全部命令
```

### Git / 推送到 GitHub

远程仓库：`origin` → https://github.com/aidenz0/astro-blog.git（默认分支 `main`，本地 `main` 跟踪 `origin/main`）。

```bash
# 日常发布流程：写完文章 / 改完代码后
git status                          # 查看改动
git add -A                          # 暂存全部改动（或 git add <文件>）
git commit -m "Add xxx 文章"        # 提交，写清楚做了什么
git push                            # 推送到 origin/main（已设上游，可省略远程和分支）
# 等价写法：
git push origin main

# 首次克隆 / 换机器
git clone https://github.com/aidenz0/astro-blog.git
cd astro-blog && pnpm install

# 拉取远程最新（多设备写作时，动手前先同步）
git pull --rebase                   # 拉取并把本地提交叠到最新之上，历史更干净

# 常用辅助
git log --oneline -10               # 看最近 10 条提交
git diff                            # 看未暂存的改动
git restore <文件>                  # 放弃某文件的未暂存改动
git restore --staged <文件>         # 取消暂存（保留改动）
```

> 推送后若已接 Vercel / Cloudflare Pages 等自动部署，`git push` 即触发线上构建发布。
> 构建会先跑 `astro check`：**文章缺 `title`/`pubDatetime`/`description`/`tags` 会导致部署失败**，
> 推送前最好本地 `pnpm run build` 验证一次。

## 架构

### 内容层
- `src/data/blog/` — 博客文章 Markdown 文件（支持子目录分类，如 `llm/`、`tools/`、`react/`）
- `src/content.config.ts` — 内容集合定义，使用 glob loader 加载 `src/data/blog/` 下所有 `.md` 文件（`_` 开头的文件除外）
- 文章 frontmatter 字段：
  - **必需**：`title`, `pubDatetime`, `description`, `tags`
  - **可选**：`slug`, `featured`, `draft`, `author`, `modDatetime`, `ogImage`, `canonicalURL`, `hideEditPost`, `timezone`

### 布局层
- `Layout.astro` — 基础布局（SEO、主题、字体）
- `Main.astro` — 首页布局
- `PostDetails.astro` — 文章详情页，集成阅读时间、TOC、代码复制等功能
- `AboutLayout.astro` — 关于页

### 路由层
- `index.astro` — 首页
- `posts/[...page].astro` — 博客列表分页
- `posts/[...slug]/index.astro` — 文章详情页（动态路由）
- `search.astro` — Pagefind 全文搜索
- `rss.xml.ts` — RSS 订阅

### 组件层
关键组件位于 `src/components/`：
- `Header.astro`, `Footer.astro` — 全局导航
- `TableOfContents.astro` — 文章目录（从 h2-h4 生成）
- `BlogStats.astro` — 博客统计（文章数、标签数、总字数）
- `Card.astro`, `Tag.astro` — 内容卡片和标签
- `ShareLinks.astro` — 社交分享
- `EditPost.astro` — 编辑页面链接（基于 `src/config.ts` 中的 `editPost` 配置）

### 工具层
- `src/utils/getSortedPosts.ts` — 文章排序（按发布时间）
- `src/utils/readingTime.ts` — 阅读时间估算
- `src/utils/slugify.ts` — URL slug 生成
- `src/utils/getPath.ts` — 路径处理

### Markdown 增强
- **数学公式**: remark-math + rehype-katex（行内 `$...$`，块级 `$$...$$`）
- **代码高亮**: Shiki（light: min-light / dark: night-owl 主题）
- **目录**: remark-toc（可折叠，标题为 "Table of contents"）
- **文件名**: @shikijs/transformers 的 transformerFileName
- **代码变换**: 支持 diff 标记、行高亮、词高亮
- **Obsidian 图片**: `src/utils/remark/remarkObsidianImages.js` — 构建时把 Obsidian 嵌入语法 `![[图片.png]]` / `![[图片.png|别名]]` 转成同目录相对路径的标准图片，交给 Astro 图片管线优化（详见 `OBSIDIAN.md`）
- **搜索**: Pagefind（构建时生成索引，搜索结果包含在 `public/pagefind/`）

### 配置
- `src/config.ts` — 站点配置（作者、标题、分页数量、时区、Edit Page 链接等）
- `astro.config.ts` — Astro 主配置（Tailwind、sitemap、Markdown 插件、Shiki）
- TypeScript 路径别名: `@/*` → `./src/*`

## 添加新文章

在 `src/data/blog/` 目录（或其子目录）下创建 `.md` 文件，frontmatter 示例：

```yaml
---
title: 文章标题
author: Aidenz
pubDatetime: 2026-05-01T05:00:00Z
slug: article-slug  # 可选，默认从文件名生成
featured: false
draft: false
tags:
  - Python
  - 异步编程
description: 文章描述摘要
---
```

文章 URL 路径会包含子目录，例如 `src/data/blog/react/xxx.md` 的 URL 为 `/posts/react/xxx`。

### 用 Obsidian 写作

推荐用 Obsidian 打开 `src/data/blog/` 作为 Vault 写作，工作流详见仓库根的 `OBSIDIAN.md`：
- 用 Templater 模板（`src/data/blog/_templates/_new-post.md`）一键生成 frontmatter；
- 图片直接粘贴存到同目录，`![[...]]` 嵌入会在构建时自动转换；
- 草稿用 `draft: true`（保留但不显示）或文件名加 `_` 前缀（loader 完全忽略）。

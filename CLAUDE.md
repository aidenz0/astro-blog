# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 AstroPaper 模板的个人技术博客，使用 Astro 6 + TailwindCSS 4 构建。博客内容存储在 `src/data/blog/` 目录下，支持中文���章、数学公式（KaTeX）、代码高亮和全文搜索。

## 命令

```bash
# 开发
pnpm run dev          # 启动开发服务器 (localhost:4321)

# 构建
pnpm run build        # 类型检查 → 构建 → 生成搜索索引
pnpm run preview      # 预览构建产物

# 代码质量
pnpm run format:check # Prettier 格式检查
pnpm run format       # Prettier 格式化
pnpm run lint         # ESLint 检查
pnpm run sync         # 生成 Astro 模块类型定义
```

## 架构

### 内容层
- `src/data/blog/` — 博客文章 Markdown 文件（支持中文、frontmatter 元数据）
- `src/content.config.ts` — 内容集合定义，使用 glob loader 加载博客
- 文章 frontmatter 必需字段：`title`, `pubDatetime`, `description`, `tags`

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

### 工具层
- `src/utils/getSortedPosts.ts` — 文章排序（按发布时间）
- `src/utils/readingTime.ts` — 阅读时间估算
- `src/utils/slugify.ts` — URL slug 生成
- `src/utils/getPath.ts` — 路径处理

### Markdown 增强
- **数学公式**: remark-math + rehype-katex
- **代码高亮**: Shiki（min-light / night-owl 主题）
- **目录**: remark-toc（可折叠）
- **文件名**: @shikijs/transformer transformerFileName
- **搜索**: Pagefind（构建时生成索引）

### 配置
- `src/config.ts` — 站点配置（作者、标题、分页数量、时区等）
- `astro.config.ts` — Astro 主配置（Tailwind、sitemap、字体）
- TypeScript 路径别名: `@/*` → `./src/*`

## 添加新文章

在 `src/data/blog/` 目录下创建 `.md` 文件，frontmatter 示例：

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

支持数学公式（`$...$` 行内，`$$...$$` 块级）和标准 Markdown 语法。

---
title: <% tp.file.title %>
author: Aidenz
pubDatetime: <% tp.date.now("YYYY-MM-DDTHH:mm:ss[Z]") %>
slug:
featured: false
draft: true
tags:
  -
description:
---

<%*
/*
  Templater 模板 —— 新建博客文章用。
  使用方式（Templater 插件）：
    1. Settings → Templater → Template folder location 设为 `_templates`
    2. 命令面板 → Templater: Create new note from template → 选本模板
  说明：
    - 本文件以 `_` 开头，glob loader (`**/[^_]*.md`) 会忽略它，不会被当成文章发布。
    - draft 默认 true；写完发布前改成 false。
    - tags / description 为必填（content.config.ts schema），留空会导致 `astro check` 构建失败。
    - 图片：直接从剪贴板粘贴即可，Obsidian 会存到同目录，
      remark-obsidian-images 插件会在构建时把 ![[图片]] 转成可优化的相对路径。
*/
-%>

# Obsidian + Astro 博客工作流（方案 A：本地图片 + 构建时转换）

用 Obsidian 写作和管理，用 Astro 构建发布，**一份 `.md` 源文件两边通吃**。

## 一次性配置

### 1. Obsidian Vault
- Vault 根 = **仓库根目录**（`.obsidian/` 在此，已加入 `.gitignore`，工作区配置不进仓库）。
- 文章写在 `src/data/blog/` 下（按子目录分类）。Vault 根虽是仓库根，但只有 `src/data/blog/` 里的 `.md` 会被 Astro 当成文章。

### 2. 附件（图片）设置 —— 关键
Obsidian → 设置 → **文件与链接**：
- **附件默认位置**：选「**与当前文件相同的文件夹**」。
  > 方案 A 约定：图片和笔记同目录，构建插件据此解析为 `./文件名`。
- 建议**关闭**「使用 `[[Wikilinks]]`」，让粘贴的图片直接生成标准 `![](xxx.png)`。
  即便忘了关、生成了 `![[xxx.png]]`，构建插件也会自动转换（见下）。

### 3. 文章模板（自动 frontmatter）
- 模板文件：[`Templates/文章模板.md`](Templates/文章模板.md)。
- 模板插件（核心 **Templates** 或社区 **Templater**）的「模板文件夹位置」设为仓库根的 `Templates`。
- 新建文章：新建空笔记 → 命令面板「插入模板」→ 选「文章模板」，再填字段。
- 系列文章：取消注释模板里的 `series` / `seriesOrder`（详见模板内注释和文末「系列文章」）。

### 4. （可选）Obsidian Git 插件
在 Obsidian 内一键 commit / push，写完直接发，不用切终端。

## 日常写作流程

1. 用模板新建文章（自动填好 frontmatter，`draft: true`）。
2. 写正文，**直接粘贴截图**——图片自动存到同目录。
3. 填好 `tags` 和 `description`（必填，否则构建失败）。
4. 本地预览：`pnpm run dev`（localhost:4321）。
5. 发布：把 `draft: true` 改成 `false` → `git commit` → `git push` → CI 自动构建上线。

## 图片转换原理

构建时 `remark-obsidian-images` 插件（[`src/utils/remark/remarkObsidianImages.js`](src/utils/remark/remarkObsidianImages.js)）会把 Obsidian 嵌入语法转成标准图片节点，交给 Astro 图片管线优化：

| Obsidian 写法 | 转换结果 |
|--------------|---------|
| `![[Pasted image 20260228.png]]` | `![](./Pasted image 20260228.png)` |
| `![[diagram.png\|架构图]]` | `![架构图](./diagram.png)`（别名→alt） |
| `![[pic.png\|400]]` | `![](./pic.png)`（纯数字尺寸忽略） |
| `[[note]]`、非图片嵌入 | 原样保留 |

> ⚠️ 图片文件必须真实存在于同目录，否则 `astro check` / 构建会因找不到图片而失败。
> 这是一道好闸门：能防止把缺图的文章发出去。

## 注意事项

- **必填字段**：`title` / `pubDatetime` / `description` / `tags`。缺任一项 `astro check` 报错、构建失败。
- **草稿**：`draft: true`（保留在仓库、列表不显示）或文件名加 `_` 前缀（loader 完全忽略）。
- **分类**：用子目录，如 `llm/`、`agent/`、`python/`。URL 会带上子目录路径。
- **`_releases/` 提示**：该目录是 AstroPaper 模板自带的发布说明，**目前会被当成文章发布**
  （loader 的 `_` 只忽略以 `_` 开头的文件名，不忽略目录）。如不想发布，给每个文件名加 `_` 前缀，
  或从仓库移除。
- **`Templates/` 安全**：模板目录在仓库根、不在 `src/data/blog/` 下，Astro 的 glob loader（base 为 `src/data/blog`）不会扫描它，不会被发布。

## 系列文章（Series）

把多篇文章组织成有序专题，在 `/series` 展示，文章页顶部显示系列导航。在 frontmatter 加两个字段：

```yaml
series: React 进阶 Pattern    # 系列名：同一系列每篇必须完全一致（按它生成 /series/<slug> 并分组，写错字会拆成两个系列）
seriesOrder: 2                # 系列内顺序 1、2、3…；不写则排到末尾按发布时间
```

- 普通文章**不写**这两个字段即可，互不影响。
- 不要留空 `seriesOrder:`（空值=null，过不了校验导致构建失败）；不需要就整行删掉。

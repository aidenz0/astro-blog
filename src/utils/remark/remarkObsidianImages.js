/**
 * remarkObsidianImages
 *
 * 把 Obsidian 的图片嵌入语法转换成标准 Markdown 图片节点，
 * 使 Astro 的图片管线能够优化它们：
 *
 *   ![[Pasted image 20260228.png]]        -> ![](./Pasted image 20260228.png)
 *   ![[diagram.png|架构图]]               -> ![架构图](./diagram.png)
 *   ![[diagram.png|400]]                  -> ![](./diagram.png)   (尺寸暂忽略，alt 留空)
 *
 * 约定（方案 A）：附件与笔记同目录，因此统一解析为同目录相对路径 `./文件名`。
 * 非图片扩展名的 ![[...]] 嵌入会原样保留，交给后续插件/渲染处理。
 *
 * 零依赖：手动遍历 mdast，不引入 unist-util-visit。
 */

const IMAGE_EXT = /\.(png|jpe?g|gif|svg|webp|avif|bmp)$/i;
// 捕获一行文本里所有 ![[ ... ]] 嵌入
const EMBED = /!\[\[([^\]]+?)\]\]/g;

/** 把 "file.png|alt或尺寸" 解析为 { url, alt } */
function parseTarget(raw) {
  const [targetRaw, aliasRaw] = raw.split("|");
  const target = targetRaw.trim();
  const alias = (aliasRaw ?? "").trim();
  // 纯数字的别名是 Obsidian 的尺寸标记，不当作 alt
  const alt = alias && !/^\d+(x\d+)?$/.test(alias) ? alias : "";
  return { url: `./${target}`, alt };
}

/** 把一个 text 节点按 ![[...]] 切成 text/image 节点数组；无匹配返回 null */
function splitTextNode(node) {
  const value = node.value;
  EMBED.lastIndex = 0;
  if (!EMBED.test(value)) return null;

  const out = [];
  let lastIndex = 0;
  EMBED.lastIndex = 0;
  let match;
  while ((match = EMBED.exec(value)) !== null) {
    const inner = match[1];
    const targetRaw = inner.split("|")[0].trim();
    // 非图片嵌入：跳过，保持原样
    if (!IMAGE_EXT.test(targetRaw)) continue;

    if (match.index > lastIndex) {
      out.push({ type: "text", value: value.slice(lastIndex, match.index) });
    }
    const { url, alt } = parseTarget(inner);
    out.push({ type: "image", url, alt, title: null });
    lastIndex = match.index + match[0].length;
  }

  if (out.length === 0) return null; // 只有非图片嵌入
  if (lastIndex < value.length) {
    out.push({ type: "text", value: value.slice(lastIndex) });
  }
  return out;
}

/** 递归遍历，替换 parent.children 中的 text 节点 */
function walk(node) {
  if (!node || !Array.isArray(node.children)) return;
  const next = [];
  for (const child of node.children) {
    if (child.type === "text") {
      const replaced = splitTextNode(child);
      if (replaced) {
        next.push(...replaced);
        continue;
      }
    }
    walk(child);
    next.push(child);
  }
  node.children = next;
}

export default function remarkObsidianImages() {
  return tree => {
    walk(tree);
  };
}

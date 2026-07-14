/**
 * remarkMermaid
 *
 * 把 ```mermaid 代码块转换成一段原始 HTML：
 *
 *   ```mermaid
 *   flowchart LR
 *     A --> B
 *   ```
 *   ->
 *   <pre class="mermaid" data-mermaid="<base64 源码>"><code>flowchart LR
 *     A --&gt; B</code></pre>
 *
 * 变成 `html` 节点后，Astro 的 Shiki 高亮不会再处理它（Shiki 只处理 `code` 节点），
 * 交给客户端脚本 `src/scripts/mermaid.ts` 在浏览器里用 mermaid.js 渲染成 SVG。
 *
 * - `data-mermaid` 存 base64(utf-8) 的原始图定义，客户端每次都从这里重新渲染，
 *   因此支持明/暗主题切换时重渲染（渲染后 SVG 会替换掉内部内容，但源码始终留在属性里）。
 * - `<code>` 里放转义后的源码，作为「未启用 JS / 渲染失败」时的兜底展示。
 *
 * 零依赖：手动遍历 mdast，不引入 unist-util-visit。
 */

/** HTML 转义，供 <code> 兜底文本使用 */
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** base64(utf-8) 编码，供 data-mermaid 属性使用 */
function toBase64Utf8(s) {
  return Buffer.from(s, "utf8").toString("base64");
}

/** 递归遍历，把 mermaid 代码块替换为 html 节点 */
function walk(node) {
  if (!node || !Array.isArray(node.children)) return;
  for (let i = 0; i < node.children.length; i++) {
    const child = node.children[i];
    if (
      child.type === "code" &&
      (child.lang || "").toLowerCase() === "mermaid"
    ) {
      const src = child.value || "";
      const html =
        `<pre class="mermaid" data-mermaid="${toBase64Utf8(src)}">` +
        `<code>${escapeHtml(src)}</code></pre>`;
      node.children[i] = { type: "html", value: html };
      continue;
    }
    walk(child);
  }
}

export default function remarkMermaid() {
  return tree => {
    walk(tree);
  };
}

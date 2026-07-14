/**
 * Mermaid 客户端渲染
 *
 * 配合 remark 插件 `src/utils/remark/remarkMermaid.js`：页面里每个 mermaid 图都是
 *   <pre class="mermaid" data-mermaid="<base64 源码>"><code>…兜底源码…</code></pre>
 *
 * 本脚本懒加载 mermaid.js（仅当页面存在 .mermaid 时才下载这个较大的库），
 * 在 `astro:page-load`（首次加载 + 每次 View Transitions 导航都会触发）时渲染，
 * 并用 MutationObserver 监听 <html data-theme> 的变化，在明/暗切换时按新主题重渲染。
 */

type MermaidApi = typeof import("mermaid").default;

let mermaidPromise: Promise<MermaidApi> | null = null;
let seq = 0;

function loadMermaid(): Promise<MermaidApi> {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then(m => m.default);
  }
  return mermaidPromise;
}

function currentTheme(): "dark" | "default" {
  return document.documentElement.getAttribute("data-theme") === "dark"
    ? "dark"
    : "default";
}

/** 解码 base64(utf-8) */
function decode(b64: string): string {
  try {
    const bin = atob(b64);
    const bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  } catch {
    return "";
  }
}

async function renderAll(): Promise<void> {
  const nodes = Array.from(
    document.querySelectorAll<HTMLElement>("pre.mermaid[data-mermaid]")
  );
  if (nodes.length === 0) return;

  const mermaid = await loadMermaid();
  mermaid.initialize({
    startOnLoad: false,
    theme: currentTheme(),
    securityLevel: "strict",
    fontFamily: "inherit",
  });

  for (const el of nodes) {
    const src = decode(el.dataset.mermaid || "");
    if (!src) continue;
    try {
      const { svg } = await mermaid.render(`mmd-${Date.now()}-${seq++}`, src);
      el.innerHTML = svg;
      el.setAttribute("data-processed", "true");
    } catch (err) {
      // 渲染失败时保留 <code> 兜底源码，方便排查
      // eslint-disable-next-line no-console
      console.error("[mermaid] render failed:", err);
    }
  }
}

// 首次加载 + 每次视图切换导航后渲染
document.addEventListener("astro:page-load", () => {
  void renderAll();
});

// 明/暗主题切换时按新主题重渲染（从 data-mermaid 源码重建，幂等）
const themeObserver = new MutationObserver(mutations => {
  if (mutations.some(m => m.attributeName === "data-theme")) {
    void renderAll();
  }
});
themeObserver.observe(document.documentElement, { attributes: true });

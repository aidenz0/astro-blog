---
title: HTML 基础完全指南 - 从零开始构建网页
author: Aidenz
pubDatetime: 2026-05-02T05:00:00Z
slug: html-basics-guide
featured: false
draft: false
tags:
  - HTML
  - 前端
  - 入门教程
description: 全面系统地介绍 HTML 基础知识，包括文本标签、属性、表单、区块等核心概念，适合初学者查缺补漏。
---

## 引言

HTML（HyperText Markup Language，超文本标记语言）是构建网页的基础语言。它使用标记标签来描述网页的结构和内容。本文将全面介绍 HTML 的核心知识点。

## 一、HTML 基础结构

### 1.1 文档结构

每个 HTML 文档都有固定的基本结构：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>页面标题</title>
  </head>
  <body>
    <!-- 页面内容 -->
  </body>
</html>
```

**结构说明：**

| 标签 | 作用 |
|------|------|
| `<!doctype html>` | 声明文档类型为 HTML5 |
| `<html>` | 根元素，lang 属性指定页面语言 |
| `<head>` | 包含文档的元数据，不可见 |
| `<meta charset="UTF-8" />` | 声明字符编码，防止中文乱码 |
| `<meta name="viewport">` | 响应式设计必需，控制移动端布局 |
| `<title>` | 页面标题，显示在浏览器标签页 |
| `<body>` | 包含页面可见内容 |

## 二、常见文本标签

### 2.1 标题标签

HTML 提供了 6 级标题，从 `<h1>` 到 `<h6>`，重要性依次递减：

```html
<h1>一级标题 - 页面主标题</h1>
<h2>二级标题 - 章节标题</h2>
<h3>三级标题 - 小节标题</h3>
<h4>四级标题</h4>
<h5>五级标题</h5>
<h6>六级标题</h6>
```

**使用原则：**
- 每个页面只有一个 `<h1>`
- 标题级别按顺序嵌套，不要跳跃
- 标题用于结构，不是用来控制字体大小

### 2.2 段落与文本格式

```html
<!-- 段落 -->
<p>这是一个段落，会自动在上下添加间距。</p>

<!-- 文本格式化标签 -->
<b>粗体文本（仅样式）</b>
<strong>粗体文本（表示重要性）</strong>

<i>斜体文本（仅样式）</i>
<em>斜体文本（表示强调）</em>

<u>下划线文本</u>
<s>删除线文本</s>
<del>表示删除的内容</del>

<mark>高亮标记文本</mark>
<small>小号文本</mark>
<sub>下标</sub>
<sup>上标</sup>

<ins>插入的文本</ins>
```

### 2.3 列表

**无序列表**（项目符号）：

```html
<ul>
  <li>无序列表项 1</li>
  <li>无序列表项 2</li>
  <li>无序列表项 3</li>
</ul>
```

**有序列表**（数字编号）：

```html
<ol>
  <li>第一步</li>
  <li>第二步</li>
  <li>第三步</li>
</ol>
```

**自定义列表**（术语解释）：

```html
<dl>
  <dt>HTML</dt>
  <dd>超文本标记语言</dd>
  <dt>CSS</dt>
  <dd>层叠样式表</dd>
</dl>
```

### 2.4 表格

```html
<table border="1">
  <thead>
    <tr>
      <th>表头 1</th>
      <th>表头 2</th>
      <th>表头 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>单元格 1</td>
      <td>单元格 2</td>
      <td>单元格 3</td>
    </tr>
    <tr>
      <td>单元格 4</td>
      <td>单元格 5</td>
      <td>单元格 6</td>
    </tr>
  </tbody>
</table>
```

**表格标签说明：**

| 标签 | 作用 |
|------|------|
| `<table>` | 表格容器 |
| `<tr>` | 表格行 |
| `<th>` | 表头单元格（加粗居中） |
| `<td>` | 普通单元格 |
| `<thead>` | 表头区域 |
| `<tbody>` | 表格主体 |
| `<tfoot>` | 表格尾部 |

**合并单元格：**

```html
<!-- 跨列合并 -->
<td colspan="2">占两列</td>

<!-- 跨行合并 -->
<td rowspan="2">占两行</td>
```

### 2.5 分隔与特殊字符

```html
<!-- 水平线 -->
<hr />

<!-- 换行 -->
<br />

<!-- 空格（多个空格只显示一个，需用实体） -->
&nbsp;&nbsp;&nbsp;

<!-- 常用 HTML 实体 -->
&lt;    <!-- 小于号 < -->
&gt;    <!-- 大于号 > -->
&amp;   <!-- 和号 & -->
&quot;  <!-- 双引号 " -->
&copy;  <!-- 版权符号 © -->
```

## 三、HTML 属性

属性为 HTML 元素提供附加信息，格式为 `name="value"`。

### 3.1 全局属性

所有元素都可以使用的属性：

```html
<!-- id：唯一标识符 -->
<div id="header">页面头部</div>

<!-- class：类名（可多个） -->
<div class="container main">内容区域</div>

<!-- style：内联样式 -->
<p style="color: red; font-size: 16px;">红色文字</p>

<!-- title：鼠标悬停提示 -->
<span title="这是提示信息">悬停查看</span>

<!-- hidden：隐藏元素 -->
<div hidden>这个元素不可见</div>

<!-- data-*：自定义数据属性 -->
<div data-id="123" data-name="test">自定义数据</div>
```

### 3.2 链接属性

```html
<!-- href：链接目标地址 -->
<a href="https://www.aidenz.eu.org">访问我的网站</a>
<a href="about.html">关于页面</a>
<a href="#section1">跳转到页面内锚点</a>
<a href="mailto:example@email.com">发送邮件</a>

<!-- target：链接打开方式 -->
<a href="https://www.example.com" target="_blank">在新标签页打开</a>
<a href="https://www.example.com" target="_self">在当前标签页打开（默认）</a>
<a href="https://www.example.com" target="_parent">在父框架中打开</a>
<a href="https://www.example.com" target="_top">在顶层框架中打开</a>

<!-- download：下载文件 -->
<a href="file.pdf" download>下载 PDF</a>
```

**target 属性值说明：**

| 值 | 说明 |
|----|------|
| `_blank` | 新窗口/标签页打开 |
| `_self` | 当前窗口（默认） |
| `_parent` | 父框架 |
| `_top` | 顶层框架 |

### 3.3 图片属性

```html
<!-- 基本用法 -->
<img src="image.jpg" alt="图片描述" />

<!-- 完整属性 -->
<img
  src="https://example.com/image.jpg"
  alt="图片无法加载时显示的文字"
  width="300"
  height="200"
  title="鼠标悬停提示"
  loading="lazy"
/>

<!-- srcset：响应式图片 -->
<img
  src="small.jpg"
  srcset="small.jpg 320w, medium.jpg 640w, large.jpg 1024w"
  sizes="(max-width: 600px) 320px, 640px"
  alt="响应式图片"
/>
```

**重要属性说明：**

| 属性 | 必需 | 说明 |
|------|------|------|
| `src` | 是 | 图片路径（本地或 URL） |
| `alt` | 是 | 替代文本，对 SEO 和无障碍访问很重要 |
| `width/height` | 否 | 设置尺寸，只设一个会按比例缩放 |
| `loading` | 否 | `"lazy"` 实现懒加载优化性能 |

### 3.4 路径说明

```html
<!-- 绝对路径（完整 URL） -->
<img src="https://www.example.com/images/logo.png" />
<a href="https://www.aidenz.eu.org">链接</a>

<!-- 相对路径 -->
<img src="logo.png" />                    <!-- 同目录 -->
<img src="./images/logo.png" />            <!-- 子目录 -->
<img src="../images/logo.png" />           <!-- 上级目录的子目录 -->
<a href="about.html">同目录页面</a>

<!-- 根相对路径（从网站根目录开始） -->
<img src="/assets/images/logo.png" />
```

## 四、HTML 表单

表单用于收集用户输入，是网页交互的核心。

### 4.1 form 标签

```html
<form action="/submit" method="POST" enctype="multipart/form-data">
  <!-- 表单内容 -->
</form>
```

**form 属性说明：**

| 属性 | 说明 |
|------|------|
| `action` | 表单提交的目标 URL |
| `method` | 提交方式：GET 或 POST |
| `enctype` | 编码类型，上传文件需设为 `multipart/form-data` |
| `target` | 提交后的响应显示位置 |
| `autocomplete` | 是否启用自动完成 |
| `novalidate` | 提交时不验证 |

### 4.2 input 输入框

```html
<!-- 文本输入 -->
<input type="text" placeholder="请输入用户名" />
<input type="password" placeholder="请输入密码" />

<!-- 带标签的输入框（推荐） -->
<label for="username">用户名</label>
<input type="text" id="username" placeholder="请输入用户名" />

<!-- 另一种写法：嵌套 -->
<label>
  密码
  <input type="password" id="password" placeholder="请输入密码" />
</label>
```

**input 常用属性：**

| 属性 | 说明 |
|------|------|
| `type` | 输入类型 |
| `id` | 唯一标识，关联 label |
| `name` | 提交时的字段名 |
| `value` | 默认值 |
| `placeholder` | 提示文本 |
| `required` | 必填 |
| `disabled` | 禁用 |
| `readonly` | 只读 |
| `maxlength` | 最大长度 |
| `minlength` | 最小长度 |
| `pattern` | 正则表达式验证 |

### 4.3 input 类型

```html
<!-- 文本相关 -->
<input type="text" />        <!-- 单行文本 -->
<input type="password" />    <!-- 密码 -->
<input type="email" />       <!-- 邮箱（自动验证） -->
<input type="url" />         <!-- URL（自动验证） -->
<input type="tel" />         <!-- 电话 -->
<input type="search" />      <!-- 搜索框 -->

<!-- 数字相关 -->
<input type="number" min="1" max="100" step="1" />
<input type="range" min="0" max="100" />

<!-- 日期时间 -->
<input type="date" />
<input type="time" />
<input type="datetime-local" />
<input type="month" />
<input type="week" />

<!-- 选择类型 -->
<input type="radio" name="sex" value="male" />男
<input type="radio" name="sex" value="female" />女

<input type="checkbox" name="hobby" value="basketball" />篮球
<input type="checkbox" name="hobby" value="football" />足球

<!-- 其他类型 -->
<input type="color" />           <!-- 颜色选择器 -->
<input type="file" />            <!-- 文件上传 -->
<input type="file" multiple />   <!-- 多文件上传 -->
<input type="hidden" />          <!-- 隐藏字段 -->
```

### 4.4 下拉选择

```html
<select name="city">
  <option value="">请选择城市</option>
  <option value="beijing">北京</option>
  <option value="shanghai">上海</option>
  <option value="guangzhou">广州</option>
</select>

<!-- 分组 -->
<select name="category">
  <optgroup label="水果">
    <option value="apple">苹果</option>
    <option value="banana">香蕉</option>
  </optgroup>
  <optgroup label="蔬菜">
    <option value="tomato">番茄</option>
    <option value="potato">土豆</option>
  </optgroup>
</select>

<!-- 多选 -->
<select name="skills" multiple size="3">
  <option value="html">HTML</option>
  <option value="css">CSS</option>
  <option value="js">JavaScript</option>
</select>
```

### 4.5 文本域

```html
<textarea
  name="message"
  rows="5"
  cols="30"
  placeholder="请输入您的留言"
  maxlength="200"
>
  默认内容
</textarea>
```

### 4.6 按钮

```html
<!-- 提交按钮 -->
<input type="submit" value="提交" />
<button type="submit">提交表单</button>

<!-- 普通按钮 -->
<input type="button" value="点击" />
<button type="button">点击我</button>

<!-- 重置按钮 -->
<input type="reset" value="重置" />
<button type="reset">重置表单</button>

<!-- 带图标的按钮 -->
<button>
  <img src="icon.png" alt="" />
  点击
</button>
```

### 4.7 表单辅助标签

```html
<!-- 字段集 -->
<fieldset>
  <legend>个人信息</legend>
  <label>姓名：<input type="text" name="name" /></label>
  <label>年龄：<input type="number" name="age" /></label>
</fieldset>

<!-- 输出结果 -->
<output name="result">计算结果</output>

<!-- 进度条 -->
<progress value="70" max="100">70%</progress>

<!-- 计量器 -->
<meter value="0.6" min="0" max="1" low="0.3" high="0.7" optimum="0.5">
  60%
</meter>
```

## 五、HTML 区块元素

### 5.1 块级元素 vs 行内元素

**块级元素（Block）**：独占一行，可设置宽高

```html
<div>通用容器</div>
<h1>~<h6>标题</h1>
<p>段落</p>
<ul><ol><li>列表</li></ol></ul>
<table>表格</table>
<form>表单</form>
<hr>水平线
<pre>预格式化文本</pre>
<blockquote>引用块</blockquote>
```

**行内元素（Inline）**：不独占一行，宽高由内容决定

```html
<span>通用容器</span>
<a>链接</a>
<img>图片
<strong><b>加粗</b></strong>
<em><i>斜体</i></em>
<br>换行
<input>表单元素
<button>按钮</button>
```

**行内块元素（Inline-block）**：不独占一行，但可设置宽高

```html
<img />
<button />
<input />
<textarea />
<select />
```

### 5.2 语义化标签

HTML5 引入了更多语义化标签，让页面结构更清晰：

```html
<body>
  <!-- 页面头部 -->
  <header>
    <h1>网站标题</h1>
    <nav>
      <a href="#">首页</a>
      <a href="#">分类</a>
      <a href="#">关于</a>
    </nav>
  </header>

  <!-- 导航区域 -->
  <nav class="breadcrumb">
    <a href="#">首页</a> > <a href="#">分类</a> > <span>当前页</span>
  </nav>

  <!-- 主要内容区域 -->
  <main>
    <article>
      <h2>文章标题</h2>
      <p>文章内容...</p>
    </article>

    <section>
      <h3>相关文章</h3>
      <ul>
        <li>文章 1</li>
        <li>文章 2</li>
      </ul>
    </section>
  </main>

  <!-- 侧边栏 -->
  <aside>
    <h3>热门标签</h3>
    <!-- 标签列表 -->
  </aside>

  <!-- 页面底部 -->
  <footer>
    <p>&copy; 2025 版权所有</p>
  </footer>
</body>
```

**语义化标签说明：**

| 标签 | 用途 |
|------|------|
| `<header>` | 页面或区块的头部 |
| `<nav>` | 导航链接区域 |
| `<main>` | 页面主要内容（每个页面只有一个） |
| `<article>` | 独立的内容单元（文章、博客） |
| `<section>` | 文档的章节 |
| `<aside>` | 侧边栏、相关内容 |
| `<footer>` | 页面或区块的底部 |
| `<figure>` | 图片、图表等独立内容 |
| `<figcaption>` | figure 的标题 |
| `<details>` | 可折叠的详细信息 |
| `<summary>` | details 的标题 |

```html
<figure>
  <img src="chart.png" alt="销售图表" />
  <figcaption>图 1：2024年销售数据</figcaption>
</figure>

<details>
  <summary>点击查看详情</summary>
  <p>这里是详细内容...</p>
</details>
```

### 5.3 div 与 span 的区别

```html
<!-- div：块级容器，用于布局 -->
<div class="nav">
  <a href="#">链接1</a>
  <a href="#">链接2</a>
</div>

<div class="content">
  <h1>标题</h1>
  <p>内容...</p>
</div>

<!-- span：行内容器，用于包裹小块内容 -->
<p>这是一段文字，<span class="highlight">这部分高亮</span>显示。</p>

<span>链接点击这里 <a href="#">链接</a></span>
```

**使用原则：**
- `div` 用于布局和结构
- `span` 用于文本样式和微调
- 优先使用语义化标签

## 六、HTML 高级特性

### 6.1 iframe 内嵌框架

```html
<iframe
  src="https://www.example.com"
  width="800"
  height="600"
  frameborder="0"
  allowfullscreen
>
  您的浏览器不支持 iframe
</iframe>

<!-- 嵌入 YouTube 视频 -->
<iframe
  src="https://www.youtube.com/embed/VIDEO_ID"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen
>
</iframe>
```

### 6.2 多媒体标签

**音频：**

```html
<audio controls>
  <source src="audio.mp3" type="audio/mpeg" />
  <source src="audio.ogg" type="audio/ogg" />
  您的浏览器不支持 audio 标签
</audio>

<!-- 自动播放、循环、静音 -->
<audio autoplay loop muted src="bgm.mp3"></audio>
```

**视频：**

```html
<video width="640" height="360" controls poster="poster.jpg">
  <source src="video.mp4" type="video/mp4" />
  <source src="video.webm" type="video/webm" />
  <track src="subtitles.vtt" kind="subtitles" srclang="zh" label="中文" />
  您的浏览器不支持 video 标签
</video>
```

### 6.3 meta 标签

```html
<head>
  <!-- 字符编码 -->
  <meta charset="UTF-8" />

  <!-- 视口设置（响应式必需） -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <!-- 页面描述（SEO） -->
  <meta name="description" content="页面描述，150字以内" />
  <meta name="keywords" content="关键词1, 关键词2" />
  <meta name="author" content="作者名" />

  <!-- 搜索引擎爬虫 -->
  <meta name="robots" content="index, follow" />
  <meta name="googlebot" content="index, follow" />

  <!-- 社交媒体分享 -->
  <meta property="og:title" content="分享标题" />
  <meta property="og:description" content="分享描述" />
  <meta property="og:image" content="分享图片URL" />
  <meta property="og:url" content="页面URL" />

  <!-- 自动刷新 -->
  <meta http-equiv="refresh" content="30;url=https://example.com" />

  <!-- IE 兼容 -->
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
</head>
```

### 6.4 安全相关

```html
<!-- 防止点击劫持 -->
<meta http-equiv="Content-Security-Policy" content="frame-ancestors 'none'" />

<!-- XSS 防护 -->
<meta http-equiv="X-XSS-Protection" content="1; mode=block" />

<!-- 链接安全属性 -->
<a href="https://untrusted.com" rel="noopener noreferrer">
  外部链接
</a>

<!-- 下载属性 -->
<a href="file.pdf" download="filename.pdf">下载 PDF</a>
```

## 七、HTML 最佳实践

### 7.1 代码规范

1. **使用小写标签名**：`<div>` 而非 `<DIV>`
2. **始终闭合标签**：自闭合标签使用 `<tag />`
3. **使用语义化标签**：用 `<header>` 而不是 `<div class="header">`
4. **添加 alt 属性**：图片必须有 alt 文本
5. **正确的嵌套**：块级元素可以包含行内元素，反之不行

### 7.2 无障碍访问

```html
<!-- 为图片提供描述性 alt -->
<img src="chart.png" alt="2024年销售增长30%" />

<!-- 为表单元素关联 label -->
<label for="email">邮箱</label>
<input type="email" id="email" />

<!-- 使用语义化标签 -->
<nav aria-label="主导航">
  <a href="#">首页</a>
</nav>

<!-- ARIA 属性（必要时） -->
<button aria-label="关闭对话框">&times;</button>
<div role="alert">操作成功！</div>
```

### 7.3 SEO 优化

```html
<!-- 每个页面一个 h1 -->
<h1>页面主标题</h1>

<!-- 正确使用标题层级 -->
<h2>章节</h2>
<h3>小节</h3>

<!-- 描述性 meta 标签 -->
<meta name="description" content="准确描述页面内容" />

<!-- 语义化结构 -->
<article>
  <header>
    <h1>文章标题</h1>
    <time datetime="2025-05-02">2025年5月2日</time>
  </header>
</article>
```

## 八、学习检查清单

掌握以下知识点表示你已打好 HTML 基础：

- [ ] HTML 基本结构和文档声明
- [ ] 文本标签（h1-h6, p, span, strong, em 等）
- [ ] 列表（ul, ol, dl）
- [ ] 表格（table, tr, th, td）
- [ ] 链接和图片（a, img）
- [ ] 表单元素（input, textarea, select, button）
- [ ] 表单属性（placeholder, required, pattern 等）
- [ ] 块级元素与行内元素的区别
- [ ] div 和 span 的正确使用
- [ ] HTML5 语义化标签
- [ ] meta 标签的作用
- [ ] 路径的写法（相对路径、绝对路径）
- [ ] HTML 实体字符

## 总结

HTML 是网页的骨架，掌握其基础标签和属性是前端开发的第一步。通过本文的学习，你应该能够：

1. 编写结构良好的 HTML 文档
2. 使用合适的标签表达内容语义
3. 创建可访问的表单
4. 理解块级元素与行内元素的区别

下一步建议学习 CSS，为你的 HTML 页面添加样式和布局。

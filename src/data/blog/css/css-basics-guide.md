---
title: CSS 基础完全指南 - 从零开始美化网页
author: Aidenz
pubDatetime: 2026-04-02T06:00:00Z
slug: css-basics-guide
featured: false
draft: false
tags:
  - CSS
  - 前端
  - 入门教程
description: 全面系统地介绍 CSS 基础知识，包括选择器、盒子模型、布局、定位等核心概念，适合初学者查缺补漏。
---

## 引言

CSS（Cascading Style Sheets，层叠样式表）用于控制网页的外观和布局。如果说 HTML 是网页的骨架，那么 CSS 就是网页的皮肤。本文将全面介绍 CSS 的核心知识点。

## 一、CSS 导入方式

### 1.1 三种导入方式

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>CSS 导入方式</title>

  <!-- 1. 外部样式表（推荐） -->
  <link rel="stylesheet" href="./css/style.css" />

  <!-- 2. 内部样式表 -->
  <style>
    p {
      color: rgb(235, 53, 207);
      font-size: 30px;
    }
    h2 {
      color: rgb(17, 224, 76);
    }
  </style>
</head>
<body>
  <!-- 3. 内联样式（不推荐） -->
  <h1 style="color: aqua;">这是一级标题，使用内联样式</h1>
  <h2>这是二级标题，使用内部样式</h2>
  <h3>这是三级标题，使用外部样式</h3>
  <p>这是段落，使用内部样式</p>
</body>
</html>
```

**三种方式对比：**

| 方式 | 优先级 | 使用场景 | 推荐度 |
|------|--------|----------|--------|
| 内联样式 | 最高 | 临时测试、动态样式 | ⭐ |
| 内部样式表 | 中 | 单页面样式、页面特有样式 | ⭐⭐ |
| 外部样式表 | 低 | 项目主样式，可复用 | ⭐⭐⭐ |

**优先级规则：** 内联 > 内部 > 外部（后面定义的会覆盖前面相同权重的样式）

### 1.2 @import 导入

```css
/* 在 CSS 文件中导入其他 CSS */
@import url("reset.css");
@import "custom.css";
```

> 注意：`@import` 会阻塞页面渲染，建议使用 `<link>` 标签。

## 二、CSS 选择器

选择器用于选中需要添加样式的 HTML 元素。

### 2.1 基础选择器

```css
/* 1. 通用选择器：选中所有元素 */
* {
  font-family: "Monaco", "Consolas", monospace;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* 2. 元素选择器：按标签名选择 */
h1 {
  color: red;
}
p {
  font-size: 16px;
}

/* 3. 类选择器：按 class 属性选择（可复用） */
.highlight {
  background-color: yellow;
}

/* 4. id 选择器：按 id 属性选择（唯一） */
#header {
  font-size: 30px;
  color: tomato;
}
```

**选择器优先级：** id > 类 > 元素 > 通用

### 2.2 关系选择器

```css
/* 1. 子代选择器：直接子元素 */
.father > .son {
  color: goldenrod;
}
.father > p {
  background-color: lightblue;
}

/* 2. 后代选择器：所有后代元素 */
.father .son {
  color: green;
}
.father p {
  background-color: rgb(217, 210, 210);
}

/* 3. 相邻兄弟选择器：下一个兄弟元素 */
h5 + p {
  background-color: ghostwhite;
}

/* 4. 通用兄弟选择器：之后所有兄弟元素 */
h5 ~ p {
  background-color: lightgray;
}
```

**关系选择器示例：**

```html
<div class="father">
  <p class="son">直接子元素</p>
  <div>
    <p class="grandson">孙代元素（后代选择器可选中）</p>
  </div>
</div>
<p>h5 的相邻兄弟</p>
<h5>标题</h5>
<p>h5 的相邻兄弟（被 + 选中）</p>
<p>h5 的兄弟（被 ~ 选中）</p>
```

### 2.3 属性选择器

```css
/* 存在指定属性 */
[target] {
  color: red;
}

/* 属性等于特定值 */
[target="_blank"] {
  color: blue;
}

/* 属性值以指定内容开头 */
[href^="https"] {
  color: green;
}

/* 属性值以指定内容结尾 */
[href$=".pdf"] {
  color: orange;
}

/* 属性值包含指定内容 */
[class*="nav"] {
  background-color: yellow;
}

/* 属性值用空格分隔，包含指定词 */
[class~="active"] {
  font-weight: bold;
}
```

### 2.4 伪类选择器

```css
/* :hover - 鼠标悬停 */
#element:hover {
  background-color: bisque;
}

/* :active - 被激活时（按下鼠标） */
button:active {
  transform: scale(0.98);
}

/* :focus - 获得焦点 */
input:focus {
  border-color: blue;
  outline: none;
}

/* :visited - 已访问的链接 */
a:visited {
  color: purple;
}

/* :link - 未访问的链接 */
a:link {
  color: blue;
}

/* :first-child - 第一个子元素 */
li:first-child {
  font-weight: bold;
}

/* :last-child - 最后一个子元素 */
li:last-child {
  border-bottom: none;
}

/* :nth-child(n) - 第 n 个子元素 */
li:nth-child(3) {
  color: red;
}
li:nth-child(odd) {
  background-color: lightgray;
}
li:nth-child(even) {
  background-color: white;
}

/* :not() - 否定选择器 */
:not(.special) {
  color: gray;
}
```

**链接伪类顺序（LoVe HAte 原则）：**

```css
/* 正确顺序 */
a:link { color: blue; }      /* 未访问 */
a:visited { color: purple; } /* 已访问 */
a:hover { color: red; }      /* 悬停 */
a:active { color: orange; }  /* 激活 */
```

### 2.5 伪元素选择器

```css
/* ::before - 在元素前插入内容 */
.box::before {
  content: "前缀：";
  color: red;
}

/* ::after - 在元素后插入内容 */
.box::after {
  content: "后缀";
  color: blue;
}

/* ::first-letter - 首字母 */
p::first-letter {
  font-size: 24px;
  color: red;
}

/* ::first-line - 第一行 */
p::first-line {
  font-weight: bold;
}

/* ::selection - 被选中的文本 */
::selection {
  background-color: yellow;
  color: black;
}
```

### 2.6 选择器优先级计算

**优先级（权重）计算规则：**

| 选择器类型 | 权重值 |
|------------|--------|
| 内联样式 | 1000 |
| id 选择器 | 100 |
| 类选择器、伪类、属性选择器 | 10 |
| 元素选择器、伪元素 | 1 |
| 通用选择器 | 0 |

```css
/* 权重：1 */
div {
  color: gray;
}

/* 权重：10 + 1 = 11 */
div.box {
  color: blue;
}

/* 权重：10 + 10 + 1 = 21 */
div.box.active {
  color: red;
}

/* 权重：100 */
#header {
  color: green;
}

/* 权重：100 + 10 = 110 */
#header.active {
  color: orange;
}

/* !important 优先级最高（慎用） */
.special {
  color: pink !important;
}
```

> 注意：`!important` 会破坏样式层叠规则，应尽量避免使用。

## 三、CSS 常用属性

### 3.1 文本属性

```css
/* 字体相关 */
font-family: "Microsoft YaHei", sans-serif;  /* 字体系列 */
font-size: 16px;                              /* 字号 */
font-weight: bold;                            /* 粗细：100-900 */
font-style: italic;                           /* 斜体 */
font-variant: small-caps;                     /* 小型大写字母 */

/* 字体简写 */
/* font: style variant weight size/line-height family */
font: italic bold 20px/1.5 "Microsoft YaHei", sans-serif;

/* 文本颜色 */
color: #ff0000;           /* 十六进制 */
color: rgb(255, 0, 0);    /* RGB */
color: rgba(255, 0, 0, 0.5); /* RGBA 带透明度 */
color: hsl(0, 100%, 50%); /* HSL */

/* 文本对齐 */
text-align: left;         /* left, center, right, justify */
text-align-last: center;  /* 最后一行对齐 */

/* 文本修饰 */
text-decoration: none;    /* none, underline, line-through */
text-decoration: underline dashed red; /* 组合写法 */

/* 文本转换 */
text-transform: uppercase; /* uppercase, lowercase, capitalize */

/* 文本阴影 */
text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);

/* 行高 */
line-height: 1.5;         /* 无单位表示倍数 */
line-height: 24px;

/* 字符/词间距 */
letter-spacing: 2px;
word-spacing: 10px;

/* 文本溢出处理 */
text-overflow: ellipsis;
white-space: nowrap;
overflow: hidden;
```

**多行文本溢出省略：**

```css
.ellipsis-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

### 3.2 背景属性

```css
/* 背景颜色 */
background-color: #ffffff;
background-color: transparent; /* 透明 */

/* 背景图片 */
background-image: url("image.jpg");
background-image: linear-gradient(to right, red, blue);
background-image: radial-gradient(circle, red, blue);

/* 背景重复 */
background-repeat: no-repeat; /* repeat, repeat-x, repeat-y, no-repeat */

/* 背景位置 */
background-position: center center;
background-position: 50% 50%;
background-position: 10px 20px;

/* 背景尺寸 */
background-size: cover;       /* 覆盖整个容器 */
background-size: contain;     /* 完整显示图片 */
background-size: 100% 100%;   /* 拉伸填充 */

/* 背景滚动 */
background-attachment: scroll; /* scroll, fixed, local */

/* 背景简写 */
/* background: color image position/size repeat attachment origin clip */
background: url("image.jpg") center/cover no-repeat fixed;

/* 多背景 */
background:
  url("bg1.png") no-repeat,
  url("bg2.png") repeat-x;
```

### 3.3 边框属性

```css
/* 边框简写：width style color */
border: 2px solid #000;

/* 单独设置各边 */
border-top: 1px solid red;
border-right: 2px dashed blue;
border-bottom: 3px dotted green;
border-left: 4px double orange;

/* 分别设置属性 */
border-width: 2px;
border-style: solid;  /* solid, dashed, dotted, double, groove, ridge, inset, outset */
border-color: #000;

/* 圆角 */
border-radius: 10px;
border-radius: 50%; /* 圆形 */
border-radius: 10px 20px 30px 40px; /* 左上 右上 右下 左下 */

/* 单独设置圆角 */
border-top-left-radius: 10px;
border-bottom-right-radius: 20px;

/* 轮廓（不占空间） */
outline: 2px solid blue;
outline-offset: 5px; /* 轮廓偏移 */
```

### 3.4 display 属性

```css
/* block - 块级元素 */
.block {
  display: block;
  width: 100px;
  height: 100px;
}

/* inline - 行内元素 */
.inline {
  display: inline;
  /* width 和 height 无效 */
}

/* inline-block - 行内块级元素 */
.inline-block {
  display: inline-block;
  width: 100px;
  height: 100px;
  /* 可设置宽高，但不独占一行 */
}

/* none - 隐藏元素 */
.hidden {
  display: none;
  /* 元素不显示，不占空间 */
}
```

**display 其他值：**

```css
/* Flex 布局 */
display: flex;

/* Grid 布局 */
display: grid;

/* 表格相关 */
display: table;
display: table-row;
display: table-cell;

/* 列表项 */
display: list-item;
```

### 3.5 显示与隐藏对比

| 属性 | 是否占空间 | 是否响应事件 | 子元素是否隐藏 |
|------|------------|--------------|----------------|
| `display: none` | 否 | 否 | 是 |
| `visibility: hidden` | 是 | 否 | 是 |
| `opacity: 0` | 是 | 是 | 否（但不可见） |

```css
.invisible {
  visibility: hidden; /* 占空间但不可见 */
}

.transparent {
  opacity: 0; /* 完全透明，但仍可交互 */
}
```

## 四、CSS 盒子模型

### 4.1 盒子模型组成

每个元素被表示为一个矩形的盒子，由四部分组成：

```
┌─────────────────────────────────────────┐
│           Margin（外边距）              │
│   ┌─────────────────────────────────┐   │
│   │      Border（边框）             │   │
│   │  ┌──────────────────────────┐   │   │
│   │  │   Padding（内边距）      │   │   │
│   │  │  ┌────────────────────┐   │   │   │
│   │  │  │  Content（内容）   │   │   │   │
│   │  │  └────────────────────┘   │   │   │
│   │  │                          │   │   │
│   │  └──────────────────────────┘   │   │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

```css
.demo {
  width: 200px;
  height: 100px;
  background-color: lightblue;

  /* 边框 */
  border: 2px solid rgb(204, 54, 172);

  /* 内边距 */
  padding: 10px;
  /* padding: 10px 20px;          上下10 左右20 */
  /* padding: 10px 20px 30px;     上10 左右20 下30 */
  /* padding: 10px 20px 30px 40px; 上10 右20 下30 左40 */

  /* 外边距 */
  margin: 20px;
}
```

### 4.2 box-sizing 属性

```css
/* content-box（默认）：width 只包含内容 */
.content-box {
  box-sizing: content-box;
  width: 200px;
  padding: 20px;
  border: 5px;
  /* 实际占位 = 200 + 20*2 + 5*2 = 250px */
}

/* border-box（推荐）：width 包含 content + padding + border */
.border-box {
  box-sizing: border-box;
  width: 200px;
  padding: 20px;
  border: 5px;
  /* 实际占位 = 200px */
}

/* 全局设置 */
* {
  box-sizing: border-box;
}
```

### 4.3 Margin 特性

```css
/* 外边距合并（垂直方向） */
.box1 {
  margin-bottom: 20px;
}
.box2 {
  margin-top: 30px;
  /* 实际间距是 30px（取较大值），不是 50px */
}

/* margin 水平居中 */
.container {
  width: 800px;
  margin: 0 auto; /* 上下0 左右自动 */
}

/* margin 负值（扩大元素） */
.box {
  margin-left: -10px;
}

/* margin auto 分配剩余空间 */
.flex-item {
  margin: auto; /* 在 flex 容器中完全居中 */
}
```

## 五、CSS 布局

### 5.1 浮动布局（Float）

```css
.father {
  width: 400px;
  background-color: lightgray;
  border: 3px solid black;
  /* 清除浮动塌陷 */
  overflow: hidden;
}

.left-son {
  width: 100px;
  height: 100px;
  background-color: lightblue;
  float: left;
}

.right-son {
  width: 100px;
  height: 100px;
  background-color: lightcoral;
  float: right;
}
```

**清除浮动的方法：**

```css
/* 方法1：overflow */
.parent {
  overflow: hidden; /* 或 auto */
}

/* 方法2：clearfix 伪类 */
.clearfix::after {
  content: "";
  display: block;
  clear: both;
}

/* 方法3：空元素（不推荐） */
.clear {
  clear: both;
}
```

**float 属性值：**

| 值 | 说明 |
|----|------|
| `left` | 左浮动 |
| `right` | 右浮动 |
| `none` | 不浮动（默认） |
| `inherit` | 继承父元素 |

### 5.2 定位布局（Position）

```css
/* 1. static（默认）：正常文档流 */
.box-normal {
  position: static;
}

/* 2. relative：相对定位（相对于自身原位置） */
.box-relative {
  position: relative;
  top: 20px;
  left: 20px;
  /* 元素仍在文档流中，原位置保留 */
}

/* 3. absolute：绝对定位（相对于最近的定位祖先） */
.box2 {
  position: relative; /* 建立定位上下文 */
}
.box-absolute {
  position: absolute;
  top: 20px;
  left: 20px;
  /* 脱离文档流，原位置不保留 */
}

/* 4. fixed：固定定位（相对于浏览器窗口） */
.box-fixed {
  position: fixed;
  top: 20px;
  left: 20px;
  /* 滚动时位置不变 */
}

/* 5. sticky：粘性定位 */
.box-sticky {
  position: sticky;
  top: 10px;
  /* 滚动到指定位置时固定 */
}
```

**定位属性：**

```css
/* 偏移属性 */
top: 10px;
right: 20px;
bottom: 30px;
left: 40px;

/* 堆叠顺序（z-index） */
.z-index-1 {
  z-index: 1;
}
.z-index-2 {
  z-index: 2; /* 值越大，层级越高 */
}
```

### 5.3 Flex 布局（推荐）

```css
/* Flex 容器 */
.container {
  display: flex;

  /* 主轴方向 */
  flex-direction: row;            /* row, row-reverse, column, column-reverse */

  /* 换行 */
  flex-wrap: nowrap;              /* nowrap, wrap, wrap-reverse */

  /* 主轴对齐 */
  justify-content: flex-start;    /* flex-start, flex-end, center, space-between, space-around, space-evenly */

  /* 交叉轴对齐 */
  align-items: center;            /* flex-start, flex-end, center, baseline, stretch */

  /* 多行对齐 */
  align-content: center;          /* flex-start, flex-end, center, space-between, space-around, stretch */

  /* 简写 */
  flex-flow: row wrap;
}

/* Flex 项目 */
.item {
  /* 扩展比例 */
  flex-grow: 1;

  /* 收缩比例 */
  flex-shrink: 0;

  /* 基础大小 */
  flex-basis: 200px;

  /* 简写 */
  flex: 1 0 200px;

  /* 单独对齐 */
  align-self: center;

  /* 排序 */
  order: 1;
}
```

**Flex 常用布局示例：**

```css
/* 水平居中 */
.center-horizontal {
  display: flex;
  justify-content: center;
}

/* 垂直居中 */
.center-vertical {
  display: flex;
  align-items: center;
}

/* 完全居中 */
.center-all {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* 左右分布 */
.space-between {
  display: flex;
  justify-content: space-between;
}

/* 三栏布局 */
.three-column {
  display: flex;
}
.three-column .main {
  flex: 1;
}
```

### 5.4 Grid 布局

```css
/* Grid 容器 */
.container {
  display: grid;

  /* 列定义 */
  grid-template-columns: 1fr 2fr 1fr;       /* 三列，比例 1:2:1 */
  grid-template-columns: 200px 1fr 100px;   /* 固定 + 弹性 */
  grid-template-columns: repeat(3, 1fr);    /* 重复定义 */
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); /* 响应式 */

  /* 行定义 */
  grid-template-rows: 100px auto;

  /* 间距 */
  gap: 20px;               /* 行列间距 */
  row-gap: 10px;
  column-gap: 20px;

  /* 区域模板 */
  grid-template-areas:
    "header header header"
    "sidebar main aside"
    "footer footer footer";
}

/* Grid 项目 */
.item {
  /* 列位置 */
  grid-column: 1 / 3;      /* 从第1条线到第3条线 */
  grid-column: span 2;     /* 跨2列 */

  /* 行位置 */
  grid-row: 1 / 3;
  grid-row: span 2;

  /* 区域 */
  grid-area: header;

  /* 对齐 */
  justify-self: start;
  align-self: center;
}
```

## 六、CSS 高级特性

### 6.1 变换（Transform）

```css
/* 2D 变换 */
.transform-2d {
  /* 平移 */
  transform: translateX(100px);
  transform: translateY(50px);
  transform: translate(100px, 50px);

  /* 缩放 */
  transform: scaleX(1.5);
  transform: scaleY(0.8);
  transform: scale(1.5);

  /* 旋转 */
  transform: rotate(45deg);

  /* 倾斜 */
  transform: skewX(30deg);
  transform: skewY(20deg);
  transform: skew(30deg, 20deg);

  /* 组合 */
  transform: translate(100px, 50px) rotate(45deg) scale(1.2);
}

/* 3D 变换 */
.transform-3d {
  perspective: 1000px;      /* 透视距离 */
  transform-style: preserve-3d;

  transform: rotateX(45deg);
  transform: rotateY(45deg);
  transform: rotateZ(45deg);
  transform: translateZ(100px);
}
```

### 6.2 过渡（Transition）

```css
/* 基本语法 */
/* transition: property duration timing-function delay; */

.button {
  background-color: blue;
  transition: background-color 0.3s ease;
}

.button:hover {
  background-color: red;
}

/* 多属性过渡 */
.card {
  transition:
    background-color 0.3s ease,
    transform 0.3s ease,
    box-shadow 0.3s ease;
}

.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
}

/* 过渡时间函数 */
ease {
  transition-timing-function: ease;      /* 慢-快-慢 */
}
linear {
  transition-timing-function: linear;   /* 匀速 */
}
ease-in {
  transition-timing-function: ease-in;  /* 慢-快 */
}
ease-out {
  transition-timing-function: ease-out; /* 快-慢 */
}
cubic-bezier {
  transition-timing-function: cubic-bezier(0.68, -0.55, 0.27, 1.55);
}
```

### 6.3 动画（Animation）

```css
/* 定义关键帧 */
@keyframes slideIn {
  0% {
    transform: translateX(-100%);
    opacity: 0;
  }
  100% {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

/* 应用动画 */
.animated {
  /* animation: name duration timing-function delay iteration-count direction fill-mode; */
  animation: slideIn 0.5s ease-out;
}

/* 循环动画 */
.loading {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* 动画属性 */
.animated {
  animation-name: slideIn;
  animation-duration: 0.5s;
  animation-timing-function: ease-out;
  animation-delay: 0.2s;
  animation-iteration-count: 3; /* 或 infinite */
  animation-direction: alternate; /* normal, reverse, alternate, alternate-reverse */
  animation-fill-mode: forwards; /* none, forwards, backwards, both */
  animation-play-state: paused; /* running, paused */
}
```

### 6.4 响应式设计

```css
/* 媒体查询 */
/* 手机 */
@media screen and (max-width: 768px) {
  .container {
    width: 100%;
    padding: 10px;
  }
  .menu {
    display: none;
  }
}

/* 平板 */
@media screen and (min-width: 769px) and (max-width: 1024px) {
  .container {
    width: 750px;
  }
}

/* 桌面 */
@media screen and (min-width: 1025px) {
  .container {
    width: 1000px;
  }
}

/* 其他媒体查询特性 */
@media print { /* 打印样式 */ }
@media screen and (orientation: portrait) { /* 竖屏 */ }
@media screen and (orientation: landscape) { /* 横屏 */ }
@media (prefers-color-scheme: dark) { /* 深色模式 */ }
```

**响应式单位：**

```css
/* 视口单位 */
width: 100vw;   /* 视口宽度 */
height: 100vh;  /* 视口高度 */
font-size: 2vw; /* 相对于视口宽度 */

/* 百分比 */
width: 50%;

/* rem/em */
font-size: 16px; /* 根元素 */
font-size: 1rem; /* 16px */
font-size: 0.875rem; /* 14px */
```

### 6.5 CSS 变量

```css
/* 定义变量 */
:root {
  --primary-color: #3498db;
  --secondary-color: #2ecc71;
  --text-color: #333;
  --bg-color: #fff;
  --spacing-unit: 8px;
}

/* 使用变量 */
.button {
  background-color: var(--primary-color);
  color: var(--text-color);
  padding: var(--spacing-unit);
}

/* 变量继承 */
.card {
  --card-bg: var(--bg-color);
  background-color: var(--card-bg);
}

/* 变量作用域 */
.dark-mode {
  --primary-color: #2ecc71;
  --text-color: #fff;
  --bg-color: #333;
}

/* 变量默认值 */
.color {
  color: var(--missing-color, #000);
}
```

## 七、CSS 最佳实践

### 7.1 命名规范

```css
/* BEM 命名法：Block__Element--Modifier */
.block { }
.block__element { }
.block--modifier { }

/* 示例 */
.card { }
.card__title { }
.card__content { }
.card--featured { }
.card--hidden { }

/* 常用前缀 */
.l-header { }   /* layout */
.c-button { }   /* component */
.u-clearfix { } /* utility */
.is-active { }  /* 状态 */
.has-error { }  /* 状态 */
```

### 7.2 代码组织

```css
/* 1. 重置样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* 2. 全局变量 */
:root {
  --color-primary: #3498db;
}

/* 3. 基础样式 */
body {
  font-family: "Microsoft YaHei", sans-serif;
  line-height: 1.6;
}

/* 4. 布局组件 */
.l-container {
  max-width: 1200px;
  margin: 0 auto;
}

/* 5. 通用组件 */
.c-button {
  /* ... */
}

/* 6. 工具类 */
.u-text-center {
  text-align: center;
}
```

### 7.3 性能优化

```css
/* 避免过度限制 */
/* 不好 */
div.nav ul li a { }

/* 更好 */
.nav-link { }

/* 避免通配符选择器 */
/* 不好 */
* { margin: 0; padding: 0; }

/* 更好 */
/* 使用 reset.css */

/* 使用简写 */
/* 不好 */
margin-top: 10px;
margin-right: 10px;
margin-bottom: 10px;
margin-left: 10px;

/* 更好 */
margin: 10px;

/* 使用 transform 代替改变位置的属性 */
/* 不好 */
top: 10px;
left: 10px;

/* 更好 */
transform: translate(10px, 10px);
```

## 八、学习检查清单

掌握以下知识点表示你已打好 CSS 基础：

- [ ] CSS 三种导入方式及优先级
- [ ] 基础选择器（元素、类、id、通用）
- [ ] 关系选择器（子代、后代、兄弟）
- [ ] 伪类选择器（hover, focus, nth-child 等）
- [ ] 伪元素选择器（before, after, first-letter 等）
- [ ] 选择器优先级计算
- [ ] 文本属性（font, color, text-align 等）
- [ ] 背景属性（background 简写）
- [ ] 边框属性（border, border-radius）
- [ ] display 属性及各种值
- [ ] 盒子模型（content, padding, border, margin）
- [ ] box-sizing 的两种模式
- [ ] Float 浮动及清除方法
- [ ] Position 定位（static, relative, absolute, fixed）
- [ ] Flex 布局
- [ ] Grid 布局
- [ ] Transform 变换
- [ ] Transition 过渡
- [ ] Animation 动画
- [ ] 响应式设计和媒体查询
- [ ] CSS 变量

## 总结

CSS 是网页的皮肤，掌握其样式和布局技术是前端开发的核心。通过本文的学习，你应该能够：

1. 使用各种选择器精确定位元素
2. 理解盒子模型和布局原理
3. 使用 Flex/Grid 实现复杂布局
4. 添加过渡和动画效果
5. 编写响应式页面

下一步建议学习 JavaScript，为你的网页添加交互功能。

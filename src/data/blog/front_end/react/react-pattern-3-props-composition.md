---
title: React 进阶 Pattern 3：Props 与组件组合 - 构建可复用的组件
author: Aidenz
pubDatetime: 2026-04-10T12:00:00Z
slug: react-pattern-3-props-composition
featured: false
draft: false
series: React 进阶 Pattern
seriesOrder: 3
tags:
  - React
  - Props
  - TypeScript
  - 组件设计
  - 前端
description: 深入理解 React Props 与组件组合模式，包括 TypeScript 接口定义、默认值、children、变体模式、组件复用策略，以及 Server/Client Component 概念。
---

## 什么是组件组合？

React 的核心思想是**组合优于继承**。与其创建一个功能庞大的组件，不如将多个小组件组合在一起。

```
好的设计 ✅                    差的设计 ❌
┌──────────────┐              ┌──────────────────┐
│ <Card>       │              │ <MegaComponent>  │
│  <Header />  │              │  header 逻辑...  │
│  <Body />    │              │  body 逻辑...    │
│  <Footer />  │              │  footer 逻辑...  │
│ </Card>      │              │  所有状态混在一起 │
└──────────────┘              └──────────────────┘
```

## 一、Props 基础

### 1.1 TypeScript 接口定义

```tsx
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'destructive';  // 联合类型
  children: React.ReactNode;  // 子元素
  onClick?: () => void;       // 可选回调
  disabled?: boolean;         // 可选布尔
  type?: 'button' | 'submit' | 'reset';
  className?: string;
}

const Button = ({
  variant = 'primary',  // 默认值
  children,
  onClick,
  disabled = false,
  type = 'button',
  className = '',
}: ButtonProps) => {
  return (
    <button
      type={type}
      className={`btn btn-${variant} ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};
```

### 1.2 Props 类型速查

| 类型 | 示例 | 说明 |
|------|------|------|
| `string` | `title: string` | 字符串 |
| `number` | `count: number` | 数字 |
| `boolean` | `isVisible: boolean` | 布尔值 |
| `() => void` | `onClick: () => void` | 无返回值的函数 |
| `(id: number) => void` | `onDelete: (id: number) => void` | 带参数的函数 |
| `React.ReactNode` | `children: React.ReactNode` | 任意可渲染内容 |
| `React.ReactNode` | `icon: React.ReactNode` | 可以传 JSX 元素 |
| `'a' \| 'b'` | `variant: 'a' \| 'b'` | 联合类型（枚举） |
| `T \| undefined` | `data?: T` | 可选属性 |
| `Record<string, string>` | `errors: Record<string, string>` | 键值对对象 |

### 1.3 默认值

```tsx
// 方式 1：参数默认值（推荐）
const Button = ({ variant = 'primary', size = 'md' }: ButtonProps) => {
  // variant 和 size 有默认值
};

// 方式 2：解构后赋值（不推荐，更啰嗦）
const Button = (props: ButtonProps) => {
  const variant = props.variant || 'primary';
  const size = props.size || 'md';
};
```

## 二、children 特殊 Prop

`children` 是 React 的特殊 prop，代表组件标签之间的内容。

### 2.1 基本用法

```tsx
interface CardProps {
  children: React.ReactNode;
  title?: string;
}

const Card = ({ children, title }: CardProps) => {
  return (
    <div className="card">
      {title && <h3>{title}</h3>}
      <div className="card-body">{children}</div>
    </div>
  );
};

// 使用
<Card title="用户信息">
  <p>姓名：张三</p>
  <p>年龄：25</p>
</Card>
```

### 2.2 多插槽模式

```tsx
interface LayoutProps {
  header: React.ReactNode;
  sidebar: React.ReactNode;
  children: React.ReactNode;  // 主内容
  footer?: React.ReactNode;
}

const Layout = ({ header, sidebar, children, footer }: LayoutProps) => {
  return (
    <div className="layout">
      <header>{header}</header>
      <div className="layout-body">
        <aside>{sidebar}</aside>
        <main>{children}</main>
      </div>
      {footer && <footer>{footer}</footer>}
    </div>
  );
};

// 使用
<Layout
  header={<h1>标题</h1>}
  sidebar={<nav>导航</nav>}
  footer={<p>版权信息</p>}
>
  <p>主内容区域</p>
</Layout>
```

## 三、变体模式（Variant Pattern）

### 3.1 基于变体的样式

```tsx
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'destructive';
  children: React.ReactNode;
  onClick?: () => void;
}

const Button = ({ variant = 'primary', children, onClick }: ButtonProps) => {
  // 根据 variant 应用不同的样式类
  const variantClass = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    destructive: 'btn-destructive',
  }[variant];

  return (
    <button className={`btn ${variantClass}`} onClick={onClick}>
      {children}
    </button>
  );
};

// 使用不同变体
<Button variant="primary">确认</Button>
<Button variant="secondary">取消</Button>
<Button variant="destructive">删除</Button>
```

### 3.2 组件变体对比

```tsx
// 错误方式 ❌：为每种变体创建独立组件
const SubmitButton = ({ children }) => <button className="btn-primary">{children}</button>;
const CancelButton = ({ children }) => <button className="btn-secondary">{children}</button>;
const DeleteButton = ({ children }) => <button className="btn-destructive">{children}</button>;

// 问题：大量重复代码，难以维护

// 正确方式 ✅：使用变体 prop
const Button = ({ variant = 'primary', children }: ButtonProps) => {
  return <button className={`btn btn-${variant}`}>{children}</button>;
};
```

## 四、组件组合实战

### 4.1 可复用的 Section 组件

```tsx
interface SectionProps {
  number: number;
  title: string;
  description: string;
  children: React.ReactNode;
}

const Section = ({ number, title, description, children }: SectionProps) => {
  return (
    <div className="tutorial-section">
      <div className="section-header">
        <span className="pattern-badge">{number}</span>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {children}
      </div>
    </div>
  );
};

// 使用
<Section number={1} title="基础状态" description="useState 的基本用法">
  <Counter />
  <Clock />
</Section>
```

### 4.2 列表组件组合

```tsx
const TodoList = ({ todos }: { todos: Todo[] }) => {
  return (
    <div className="todo-list">
      {todos.map(todo => (
        <TodoItem
          key={todo.id}
          todo={todo}
        />
      ))}
    </div>
  );
};

const TodoItem = ({ todo }: { todo: Todo }) => {
  return (
    <div className={`todo-item ${todo.completed ? 'completed' : ''}`}>
      <span>{todo.text}</span>
    </div>
  );
};
```

## 五、Server Component vs Client Component

### 5.1 Next.js 中的组件类型

Next.js App Router 默认所有组件都是 **Server Component**（服务端组件）。需要交互（事件、状态）的组件必须标记为 **Client Component**。

```tsx
// Server Component（默认）
// - 不需要 'use client'
// - 不能使用 useState, useEffect, onClick 等
// - 可以直接访问数据库、文件系统
// - 更好的性能（不发送 JS 到客户端）
const ServerPage = () => {
  return (
    <div>
      <h1>这是服务端组件</h1>
      <ClientComponent />  {/* 可以嵌套客户端组件 */}
    </div>
  );
};

// Client Component
// - 必须在文件顶部声明 'use client'
// - 可以使用所有 React Hooks
// - 可以绑定事件
'use client';
const ClientComponent = () => {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
};
```

### 5.2 何时用哪种？

| 场景 | 使用 |
|------|------|
| 纯展示，无交互 | Server Component |
| 需要 useState / useEffect | Client Component |
| 需要事件处理（onClick 等） | Client Component |
| 需要浏览器 API（localStorage 等） | Client Component |
| 数据获取（数据库、API） | Server Component |
| 组件库 / 通用 UI | 看情况，通常 Client |

### 5.3 最佳实践

```tsx
// 最佳实践：Server Component 包裹 Client Component
// layout.tsx (Server)
export default function Layout({ children }) {
  return (
    <ThemeProvider>          {/* Client Component */}
      <Dashboard />         {/* Client Component */}
    </ThemeProvider>
  );
}

// page.tsx (Server)
export default function Page() {
  return <Dashboard />;     {/* Client Component */}
}
```

## 六、路径别名

Next.js 支持路径别名，避免深层相对路径：

```tsx
// 不好 ❌
import Button from '../../../components/Button';
import { useTheme } from '../../contexts/ThemeContext';

// 好 ✅（使用 @ 别名）
import Button from '@/components/Button';
import { useTheme } from '@/contexts/ThemeContext';
```

配置在 `tsconfig.json` 中：

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

## 学习检查清单

- [ ] 能用 TypeScript 接口定义 Props
- [ ] 理解 children prop 的用法
- [ ] 掌握变体模式的设计思路
- [ ] 理解 Server Component 和 Client Component 的区别
- [ ] 知道何时使用 `'use client'`
- [ ] 能使用路径别名 `@/*`

## 下一篇

[Pattern 4: 条件渲染 - 根据状态显示不同的 UI](/posts/react/react-pattern-4-conditional-rendering)

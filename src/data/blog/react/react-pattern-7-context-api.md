---
title: "React 进阶 Pattern 7：Context API - 跨组件共享状态"
author: Aidenz
pubDatetime: 2026-05-03T16:00:00Z
slug: react-pattern-7-context-api
featured: false
draft: false
tags:
  - React
  - Context API
  - Hooks
  - 前端
description: 深入理解 React Context API，包括 createContext、Provider、useContext、自定义 Hook 封装，以及暗黑主题切换的完整实战。
---

## Props Drilling 的问题

当多层嵌套的组件需要共享同一份数据时，传统的 Props 传递会导致"Props Drilling"（逐层传递）：

```
问题场景：主题需要从 App 传到深层的 Button

App (theme)
  └─ Layout (theme)         ← 不需要 theme，只是转发
       └─ Sidebar (theme)   ← 不需要 theme，只是转发
            └─ Button (theme) ← 真正需要 theme 的组件
```

每一层都要接收和传递 `theme` prop，即使中间层根本不用它。

## 一、Context API 基础

### 1.1 三个核心概念

| 概念 | 作用 |
|------|------|
| `createContext` | 创建一个上下文对象 |
| `Provider` | 提供数据的组件 |
| `useContext` | 消费数据的 Hook |

### 1.2 创建 Context

```tsx
import { createContext, useContext } from 'react';

// 1. 定义类型
interface ThemeContextType {
  theme: string;
  toggleTheme: () => void;
}

// 2. 创建 Context（初始值为 undefined）
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);
```

### 1.3 创建 Provider

```tsx
import { useState, useCallback } from 'react';

const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [theme, setTheme] = useState('light');

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

### 1.4 消费 Context

```tsx
const ThemeToggle = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <div>
      <p>当前主题：{theme}</p>
      <button onClick={toggleTheme}>切换主题</button>
    </div>
  );
};
```

## 二、自定义 Hook 封装

### 2.1 为什么需要封装？

```tsx
// 问题：直接使用 useContext 需要处理 undefined
const { theme } = useContext(ThemeContext);
// TypeScript 报错：theme 可能是 undefined

// 每个消费者都要检查
const context = useContext(ThemeContext);
if (!context) throw new Error('必须在 Provider 内使用');
const { theme } = context;
```

### 2.2 创建 useTheme 自定义 Hook

```tsx
const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);

  if (context === undefined) {
    throw new Error('useTheme 必须在 ThemeProvider 内使用');
  }

  return context;
};
```

### 2.3 使用自定义 Hook

```tsx
// 使用起来干净简洁
const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();  // 类型安全，不会是 undefined

  return (
    <div>
      <p>当前主题：{theme}</p>
      <button onClick={toggleTheme}>切换主题</button>
    </div>
  );
};
```

## 三、完整实战：主题切换系统

### 3.1 ThemeContext.tsx

```tsx
'use client';

import { createContext, useContext, useState, useCallback } from 'react';

// 类型定义
export interface ThemeContextType {
  theme: string;
  toggleTheme: () => void;
}

// 创建 Context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// 自定义 Hook
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme 必须在 ThemeProvider 内使用');
  }
  return context;
};

// Provider 组件
export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [theme, setTheme] = useState('light');

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const newTheme = prev === 'light' ? 'dark' : 'light';
      document.body.className = newTheme;  // 应用到 DOM
      return newTheme;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

### 3.2 在布局中使用 Provider

```tsx
// layout.tsx
import { ThemeProvider } from '@/contexts/ThemeContext';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
```

### 3.3 在任意深层组件中使用

```tsx
// components/ThemeToggle.tsx（可以是任意深层）
'use client';

import { useTheme } from '@/contexts/ThemeContext';

const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="widget">
      <h3>主题切换</h3>
      <p>当前：{theme === 'light' ? '亮色' : '暗色'}</p>
      <button onClick={toggleTheme}>
        切换为 {theme === 'light' ? '暗色' : '亮色'}
      </button>
    </div>
  );
};
```

## 四、CSS 主题变量

### 4.1 定义 CSS 变量

```css
/* globals.css */
:root {
  --background: #ffffff;
  --foreground: #171717;
  --card: #f9f9f9;
  --border: #e5e5e5;
  --primary: #3b82f6;
}

.dark {
  --background: #0a0a0a;
  --foreground: #ededed;
  --card: #1a1a1a;
  --border: #333333;
  --primary: #60a5fa;
}
```

### 4.2 在组件中使用

```tsx
const Card = ({ children }: { children: React.ReactNode }) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--card)',
        color: 'var(--foreground)',
        border: '1px solid var(--border)',
      }}
    >
      {children}
    </div>
  );
};
```

## 五、多个 Context

```tsx
// 可以创建多个独立的 Context
const AuthContext = createContext<AuthType | undefined>(undefined);
const ThemeContext = createContext<ThemeType | undefined>(undefined);
const LocaleContext = createContext<LocaleType | undefined>(undefined);

// 在布局中组合
const App = ({ children }) => {
  return (
    <AuthProvider>
      <ThemeProvider>
        <LocaleProvider>
          {children}
        </LocaleProvider>
      </ThemeProvider>
    </AuthProvider>
  );
};
```

## 六、Context vs Props 传递

| 特性 | Props 传递 | Context API |
|------|-----------|-------------|
| 适用层级 | 1-2 层 | 3 层以上 |
| 类型安全 | 天然安全 | 需要自定义 Hook |
| 性能 | 只影响接收组件 | 所有消费者都会重渲染 |
| 调试 | 容易追踪 | 不容易追踪来源 |
| 复杂度 | 简单直接 | 需要额外设置 |

**何时用 Context：**

- 主题、语言、用户认证等全局状态
- 跨越多层组件的数据
- 避免 Props Drilling

**何时用 Props：**

- 父子组件之间的直接通信
- 只传递 1-2 层
- 需要明确的数据来源

## 七、常见错误

### 错误 1：忘记包裹 Provider

```tsx
// 错误 ❌：使用 useTheme 但没有 Provider
const App = () => {
  return <ThemeToggle />;
  // 报错：useTheme 必须在 ThemeProvider 内使用
};

// 正确 ✅
const App = () => {
  return (
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>
  );
};
```

### 错误 2：Provider 放置位置太低

```tsx
// 错误 ❌：Provider 在使用组件的同一层
const Page = () => {
  return (
    <ThemeProvider>
      <Header />  {/* 只有 Header 能访问 */}
    </ThemeProvider>
  );
};

// 正确 ✅：Provider 放在高层
// layout.tsx
<ThemeProvider>
  <Page />
</ThemeProvider>
```

### 错误 3：Context 值变化导致不必要的重渲染

```tsx
// 问题 ❌：每次渲染都创建新对象
const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme: () => setTheme(...) }}>
      {/* 每次渲染 value 都是新对象，所有消费者都重渲染 */}
      {children}
    </ThemeContext.Provider>
  );
};

// 解决 ✅：使用 useMemo 和 useCallback
const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');
  const toggleTheme = useCallback(() => setTheme(prev => ...), []);

  const value = useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
```

## 学习检查清单

- [ ] 理解 Props Drilling 的问题
- [ ] 掌握 createContext、Provider、useContext 三件套
- [ ] 能创建自定义 Hook 封装 Context
- [ ] 理解 CSS 变量与主题切换的配合
- [ ] 知道 Context 和 Props 的适用场景
- [ ] 了解 Context 的性能注意事项

## 下一篇

[Pattern 8: 自定义 Hooks 与性能优化](/posts/react/react-pattern-8-custom-hooks-performance)

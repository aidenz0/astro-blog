---
title: "React 进阶 Pattern 8：自定义 Hooks 与性能优化"
author: Aidenz
pubDatetime: 2026-05-03T17:00:00Z
slug: react-pattern-8-custom-hooks-performance
featured: false
draft: false
tags:
  - React
  - 自定义Hooks
  - 性能优化
  - localStorage
  - 前端
description: 深入理解 React 自定义 Hooks 的编写方法，包括泛型、SSR 安全、localStorage 持久化，以及 useMemo 和 useCallback 的性能优化策略。
---

## 什么是自定义 Hook？

自定义 Hook 是一个以 `use` 开头的 JavaScript 函数，内部可以调用其他 Hook。它的核心价值是**复用状态逻辑**。

```tsx
// 自定义 Hook：封装 localStorage 读写逻辑
function useLocalStorage<T>(key: string, initialValue: T) {
  const [value, setValue] = useState<T>(initialValue);

  // 从 localStorage 读取
  useEffect(() => {
    const stored = localStorage.getItem(key);
    if (stored) setValue(JSON.parse(stored));
  }, [key]);

  // 写入 localStorage
  const setStoredValue = (newValue: T) => {
    setValue(newValue);
    localStorage.setItem(key, JSON.stringify(newValue));
  };

  return [value, setStoredValue] as const;
}

// 使用
const [notes, setNotes] = useLocalStorage<string[]>('notes', []);
```

## 一、泛型自定义 Hook

### 1.1 泛型基础

```tsx
// 泛型让 Hook 适用于任意类型
function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  // ...
}

// 不同类型都可以使用
const [name, setName] = useLocalStorage<string>('name', '');
const [count, setCount] = useLocalStorage<number>('count', 0);
const [items, setItems] = useLocalStorage<string[]>('items', []);
const [user, setUser] = useLocalStorage<User | null>('user', null);
```

### 1.2 泛型语法解析

```tsx
function useLocalStorage<T>(key: string, initialValue: T) {
  //                                     ^^^^^^^^^^^^^^^
  // T 是类型参数，调用时由传入的值推断
  // useLocalStorage('key', 'hello') → T = string
  // useLocalStorage('key', 42) → T = number
}
```

## 二、完整的 useLocalStorage 实现

### 2.1 SSR 安全版本

```tsx
'use client';

import { useState, useEffect, useCallback } from 'react';

function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  // 状态：初始值为传入的 initialValue
  const [storedValue, setStoredValue] = useState<T>(initialValue);
  const [mounted, setMounted] = useState(false);

  // 挂载后从 localStorage 读取
  useEffect(() => {
    setMounted(true);

    try {
      const item = window.localStorage.getItem(key);
      if (item) {
        setStoredValue(JSON.parse(item));
      }
    } catch (error) {
      console.error(`读取 localStorage 错误 (key: ${key}):`, error);
    }
  }, [key]);

  // 更新函数：同时更新状态和 localStorage
  const setValue = useCallback(
    (value: T) => {
      try {
        setStoredValue(value);
        if (mounted) {
          window.localStorage.setItem(key, JSON.stringify(value));
        }
      } catch (error) {
        console.error(`写入 localStorage 错误 (key: ${key}):`, error);
      }
    },
    [key, mounted]
  );

  return [storedValue, setValue];
}
```

### 2.2 为什么需要 SSR 安全？

```
Next.js 渲染流程：

1. 服务端渲染（SSR）
   → 没有 window 对象
   → localStorage 不存在
   → 如果直接访问会报错

2. 客户端水合（Hydration）
   → React 将服务端 HTML 与客户端 JS 合并
   → 要求两边的初始渲染结果一致

3. 客户端挂载完成
   → 此时可以安全访问 localStorage
   → 在 useEffect 中读取
```

**解决方案：**

```tsx
// 方案1：mounted 标志
const [mounted, setMounted] = useState(false);
useEffect(() => {
  setMounted(true);
  // 在这里访问 localStorage
}, []);

// 方案2：typeof 检查
if (typeof window !== 'undefined') {
  // 安全访问 window
}
```

## 三、使用 useLocalStorage 实战

### 3.1 笔记组件

```tsx
const NotesWidget = () => {
  const [notes, setNotes] = useLocalStorage<string[]>('tutorial-notes', []);
  const [input, setInput] = useState('');

  // 添加笔记
  const addNote = useCallback(() => {
    if (!input.trim()) return;
    setNotes([...notes, input.trim()]);
    setInput('');
  }, [input, notes, setNotes]);

  // 清空笔记
  const clearNotes = useCallback(() => {
    setNotes([]);
  }, [setNotes]);

  // 回车提交
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      addNote();
    }
  };

  return (
    <div className="widget">
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="输入笔记，按回车保存"
      />
      <button onClick={addNote}>添加</button>
      <button onClick={clearNotes}>清空</button>

      <div className="notes-list">
        {notes.map((note, index) => (
          <div key={index} className="note-item">
            {note}
          </div>
        ))}
      </div>

      {notes.length === 0 && <p>暂无笔记</p>}
    </div>
  );
};
```

### 3.2 统计计算

```tsx
const noteStats = useMemo(() => ({
  total: notes.length,
  longNotes: notes.filter(n => n.length > 10).length,
  avgLength: notes.length > 0
    ? Math.round(notes.reduce((sum, n) => sum + n.length, 0) / notes.length)
    : 0,
}), [notes]);
```

## 四、useMemo 详解

### 4.1 作用

`useMemo` 缓存计算结果，只在依赖变化时重新计算。

```tsx
const stats = useMemo(() => {
  // 昂贵的计算
  return {
    total: items.length,
    filtered: items.filter(item => item.active).length,
  };
}, [items]);  // 只在 items 变化时重新计算
```

### 4.2 何时使用

| 场景 | 是否用 useMemo |
|------|----------------|
| 简单计算（加减乘除） | 不需要 |
| 大数组的 filter/map/reduce | 推荐 |
| 生成对象/数组作为 prop | 推荐（保持引用稳定） |
| 计算结果作为 useEffect 依赖 | 推荐 |
| 生成 JSX 元素 | 不需要 |

### 4.3 实际对比

```tsx
// 没有 useMemo：每次渲染都计算
const stats = {
  total: items.length,
  filtered: items.filter(i => i.active).length,
  sorted: [...items].sort((a, b) => a.name.localeCompare(b.name)),
};
// 即使 items 没变，也会重新创建对象、filter、sort

// 有 useMemo：只在 items 变化时计算
const stats = useMemo(() => ({
  total: items.length,
  filtered: items.filter(i => i.active).length,
  sorted: [...items].sort((a, b) => a.name.localeCompare(b.name)),
}), [items]);
// items 不变时，返回缓存的结果
```

## 五、useCallback 详解

### 5.1 作用

`useCallback` 缓存函数引用，避免每次渲染都创建新函数。

```tsx
// 没有 useCallback：每次渲染创建新函数
const handleClick = () => {
  console.log(count);
};

// 有 useCallback：函数引用稳定
const handleClick = useCallback(() => {
  console.log(count);
}, [count]);
```

### 5.2 何时使用

| 场景 | 是否用 useCallback |
|------|---------------------|
| 传递给子组件的回调 | 推荐（配合 React.memo） |
| 作为 useEffect 的依赖 | 推荐 |
| 事件处理器（onClick 等） | 通常不需要 |
| 内联函数 | 不需要 |

### 5.3 useMemo vs useCallback

```tsx
// useMemo 缓存计算结果（值）
const doubled = useMemo(() => count * 2, [count]);
// 返回：2, 4, 6, ...

// useCallback 缓存函数引用
const increment = useCallback(() => setCount(c => c + 1), []);
// 返回：同一个函数引用

// 实际上 useCallback 是 useMemo 的语法糖
const increment = useCallback(() => setCount(c => c + 1), []);
// 等价于
const increment = useMemo(() => () => setCount(c => c + 1), []);
```

## 六、性能优化原则

### 6.1 先测量，再优化

```
不要过早优化！

1. 先用 React DevTools Profiler 测量
2. 找到真正慢的组件
3. 针对性地使用 useMemo/useCallback
```

### 6.2 过度优化的代价

```tsx
// 过度优化 ❌：简单计算也用 useMemo
const fullName = useMemo(() => firstName + ' ' + lastName, [firstName, lastName]);
// 比直接写还慢！因为 useMemo 本身有开销

// 正确 ✅：直接计算
const fullName = firstName + ' ' + lastName;
```

### 6.3 常见优化场景

```tsx
// 场景1：大列表过滤
const filtered = useMemo(
  () => items.filter(item => item.active),
  [items]
);

// 场景2：传递给子组件的回调
const handleSubmit = useCallback((data) => {
  onSave(data);
}, [onSave]);

// 场景3：生成稳定的对象引用
const contextValue = useMemo(
  () => ({ theme, toggleTheme }),
  [theme, toggleTheme]
);
```

## 七、其他常用自定义 Hook

### 7.1 useDebounce

```tsx
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// 使用
const SearchInput = () => {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery) {
      fetchResults(debouncedQuery);
    }
  }, [debouncedQuery]);

  return <input value={query} onChange={e => setQuery(e.target.value)} />;
};
```

### 7.2 useToggle

```tsx
function useToggle(initial = false): [boolean, () => void] {
  const [value, setValue] = useState(initial);
  const toggle = useCallback(() => setValue(v => !v), []);
  return [value, toggle];
}

// 使用
const [isOpen, toggleOpen] = useToggle(false);
```

## 学习检查清单

- [ ] 理解自定义 Hook 的命名规则和作用
- [ ] 能使用泛型创建通用 Hook
- [ ] 理解 SSR 安全的重要性
- [ ] 掌握 useLocalStorage 的完整实现
- [ ] 理解 useMemo 和 useCallback 的区别和使用场景
- [ ] 避免过度优化

## 系列总结

通过这 8 个 Pattern，你已经掌握了 React 最核心的开发模式：

1. **useState** → 管理组件状态
2. **useEffect** → 处理副作用
3. **Props** → 组件间通信与组合
4. **条件渲染** → 根据状态展示 UI
5. **列表渲染** → 高效渲染数据
6. **表单处理** → 处理用户输入
7. **Context** → 跨组件共享状态
8. **自定义 Hooks** → 复用逻辑与性能优化

这 8 个模式覆盖了 95% 的日常开发场景。下一步可以学习 React Router、Next.js 数据获取、状态管理库（Zustand）等进阶内容。

---
title: "React 进阶 Pattern 2：useEffect - 让组件与外部世界交互"
author: Aidenz
pubDatetime: 2026-05-03T11:00:00Z
slug: react-pattern-2-useeffect
featured: false
draft: false
tags:
  - React
  - useEffect
  - Hooks
  - 前端
description: 深入理解 React useEffect Hook，涵盖副作用概念、依赖数组、清理函数、定时器、数据获取等实战场景，以及常见陷阱和最佳实践。
---

## 什么是"副作用"？

组件的主要职责是渲染 UI（纯计算：输入 props/state → 输出 JSX）。但实际应用中，组件还需要做"渲染之外的事"：

- 设置定时器
- 发起网络请求
- 操作 DOM
- 订阅外部数据

这些"渲染之外的事"就叫**副作用（Side Effect）**。`useEffect` 就是用来处理副作用的 Hook。

## 一、基本语法

```tsx
import { useEffect } from 'react';

useEffect(() => {
  // 副作用代码（在渲染后执行）

  return () => {
    // 清理代码（在下次执行前或卸载时执行）
  };
}, [依赖项]);  // 依赖数组
```

**执行时机：**

| 依赖数组 | 执行时机 |
|----------|----------|
| 不传 | 每次渲染后都执行 |
| `[]`（空数组） | 只在首次渲染后执行一次 |
| `[a, b]` | 首次渲染后 + a 或 b 变化时执行 |

## 二、实战：实时时钟

### 2.1 基本实现

```tsx
const Clock = () => {
  const [time, setTime] = useState<Date | null>(null);

  useEffect(() => {
    // 设置定时器，每秒更新时间
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);

    // 清理函数：组件卸载时清除定时器
    return () => {
      clearInterval(timer);
    };
  }, []);  // 空数组 = 只执行一次

  // 避免服务端渲染水合不匹配
  if (!time) return <div>加载中...</div>;

  return <div>当前时间：{time.toLocaleTimeString()}</div>;
};
```

### 2.2 为什么需要清理函数？

```tsx
// 没有清理的后果 ❌
useEffect(() => {
  const timer = setInterval(() => {
    setTime(new Date());
  }, 1000);
  // 没有 return 清理函数
}, []);

// 问题：
// 1. 组件卸载后定时器还在跑 → 内存泄漏
// 2. 每次重新渲染都会创建新定时器 → 多个定时器同时运行
```

**清理函数的作用：**

```
组件挂载 → useEffect 执行 → 设置定时器
组件重新渲染 → useEffect 清理旧定时器 → 设置新定时器（如果依赖变化）
组件卸载 → useEffect 清理定时器
```

### 2.3 水合不匹配问题

```tsx
// 问题 ❌：服务端和客户端渲染结果不同
const Clock = () => {
  const [time, setTime] = useState(new Date());
  // 服务端渲染时间 = T1
  // 客户端首次渲染时间 = T2（可能不同！）
  // React 报错：Hydration mismatch
};

// 解决 ✅：初始值设为 null，客户端挂载后再更新
const Clock = () => {
  const [time, setTime] = useState<Date | null>(null);

  useEffect(() => {
    setTime(new Date());  // 只在客户端执行
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  if (!time) return null;  // 服务端和客户端都返回 null，一致
  return <div>{time.toLocaleTimeString()}</div>;
};
```

## 三、依赖数组详解

### 3.1 空依赖数组

```tsx
useEffect(() => {
  // 只在组件挂载时执行一次
  console.log('组件已挂载');

  return () => {
    // 只在组件卸载时执行一次
    console.log('组件将卸载');
  };
}, []);
```

**适用场景：**
- 初始化数据获取
- 设置全局事件监听
- 启动定时器

### 3.2 有依赖数组

```tsx
const [userId, setUserId] = useState(1);

useEffect(() => {
  // userId 变化时执行
  fetchUser(userId);
}, [userId]);  // userId 变化时重新执行
```

### 3.3 没有依赖数组

```tsx
useEffect(() => {
  // 每次渲染后都执行（谨慎使用！）
  console.log('组件已渲染');
});
```

**适用场景很少**，通常用于需要每次渲染都同步的场景。

## 四、依赖数组的规则

### 4.1 React 的要求

```tsx
// React 要求：effect 中用到的所有外部值，都应该放在依赖数组中

const [count, setCount] = useState(0);
const [name, setName] = useState('张三');

useEffect(() => {
  console.log(count, name);  // 用到了 count 和 name
}, [count, name]);  // 必须都放进来
```

### 4.2 常见错误

```tsx
// 错误 ❌：遗漏依赖
const [userId, setUserId] = useState(1);

useEffect(() => {
  fetchUser(userId);  // 用到了 userId
}, []);  // 但依赖数组是空的！

// 后果：userId 变化时不会重新获取数据
// ESLint 会警告：React Hook useEffect has a missing dependency
```

```tsx
// 错误 ❌：对象/数组作为依赖（每次渲染都是新引用）
const [config, setConfig] = useState({ url: '/api' });

useEffect(() => {
  fetchData(config);  // config 每次渲染都是新对象
}, [config]);  // 每次都执行！
```

## 五、清理函数模式

### 5.1 定时器清理

```tsx
useEffect(() => {
  const timer = setInterval(() => {
    // 做些什么
  }, 1000);

  return () => clearInterval(timer);
}, []);
```

### 5.2 事件监听清理

```tsx
useEffect(() => {
  const handleResize = () => {
    console.log('窗口大小变化');
  };

  window.addEventListener('resize', handleResize);

  return () => {
    window.removeEventListener('resize', handleResize);
  };
}, []);
```

### 5.3 订阅清理

```tsx
useEffect(() => {
  const subscription = someAPI.subscribe(data => {
    setData(data);
  });

  return () => {
    subscription.unsubscribe();
  };
}, []);
```

## 六、模拟 API 请求

```tsx
const UserProfile = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;  // 防止组件卸载后更新状态

    const fetchUser = async () => {
      try {
        setLoading(true);
        // 模拟 API 请求
        await new Promise(resolve => setTimeout(resolve, 2000));

        if (!cancelled) {
          setUser({ name: '张三', age: 25 });
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError('加载失败');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchUser();

    return () => {
      cancelled = true;  // 清理：标记为已取消
    };
  }, []);

  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误：{error}</div>;
  if (!user) return <div>无数据</div>;
  return <div>{user.name}</div>;
};
```

**为什么要 `cancelled` 标志？**

```
组件挂载 → 开始请求
用户导航离开 → 组件卸载
请求完成 → 尝试 setState → 但组件已卸载！
→ React 警告：Can't perform a React state update on an unmounted component

使用 cancelled 标志：请求完成时检查组件是否还在，不在就跳过 setState
```

## 七、常见错误模式

### 错误 1：在渲染中请求数据

```tsx
// 错误 ❌：每次渲染都发请求
const Component = () => {
  const [data, setData] = useState(null);

  fetch('/api/data')  // 渲染时执行，无限循环！
    .then(res => res.json())
    .then(setData);  // setState → 重新渲染 → 再 fetch...

  return <div>{data}</div>;
};

// 正确 ✅：放在 useEffect 中
const Component = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData);
  }, []);  // 只执行一次

  return <div>{data}</div>;
};
```

### 错误 2：在 effect 中没有清理副作用

```tsx
// 错误 ❌
useEffect(() => {
  window.addEventListener('resize', handleResize);
  // 没有清理！每次渲染都添加新的监听器
}, []);

// 正确 ✅
useEffect(() => {
  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);
```

### 错误 3：依赖数组不完整

```tsx
// 错误 ❌
const [a, setA] = useState(1);
const [b, setB] = useState(2);

useEffect(() => {
  console.log(a + b);  // 用到了 a 和 b
}, [a]);  // 只写了 a，b 变化时不执行

// 正确 ✅
useEffect(() => {
  console.log(a + b);
}, [a, b]);  // 都写上
```

## 学习检查清单

- [ ] 理解什么是副作用
- [ ] 掌握 useEffect 的三种依赖数组形式
- [ ] 理解清理函数的作用和时机
- [ ] 能处理水合不匹配问题
- [ ] 掌握依赖数组的规则
- [ ] 了解异步请求中的取消模式

## 下一篇

[Pattern 3: Props 与组件组合 - 让组件之间传递数据和复用 UI](/posts/react/react-pattern-3-props-composition)

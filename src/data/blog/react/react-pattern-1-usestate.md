---
title: "React 进阶 Pattern 1：useState - 让组件拥有记忆"
author: Aidenz
pubDatetime: 2026-05-03T10:00:00Z
slug: react-pattern-1-usestate
featured: false
draft: false
tags:
  - React
  - useState
  - Hooks
  - 前端
description: 深入理解 React useState Hook，涵盖基本类型、对象、数组状态管理，函数式更新，以及不可变更新模式，配合计数器实战案例。
---

## 为什么需要 useState？

普通 JavaScript 变量变化时，**页面不会自动更新**。React 需要一个机制来：

1. 保存组件的"状态"数据
2. 数据变化时自动重新渲染 UI

这就是 `useState` 的作用——**让函数组件拥有"记忆"**。

## 一、基本用法

### 1.1 声明状态

```tsx
import { useState } from 'react';

const Counter = () => {
  // 语法：const [值, 更新函数] = useState(初始值)
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>当前计数：{count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
      <button onClick={() => setCount(count - 1)}>-1</button>
      <button onClick={() => setCount(0)}>重置</button>
    </div>
  );
};
```

### 1.2 为什么普通变量不行？

```tsx
// 错误写法 ❌
const Counter = () => {
  let count = 0;  // 普通变量

  const handleClick = () => {
    count = count + 1;  // 值确实变了
    console.log(count);  // 1, 2, 3...
    // 但是页面不会更新！因为 React 不知道 count 变了
  };

  return (
    <div>
      <p>{count}</p>  {/* 永远显示 0 */}
      <button onClick={handleClick}>+1</button>
    </div>
  );
};
```

**区别对比：**

| 特性 | 普通变量 | useState |
|------|----------|----------|
| 值变化 | 是 | 是 |
| 触发重新渲染 | 否 | 是 |
| 组件重新执行后保持值 | 否（重置为初始值） | 是（React 保存） |
| 适用场景 | 临时计算 | 需要展示在 UI 上的数据 |

### 1.3 工作原理

```
首次渲染：
  useState(0) → count = 0 → 渲染 UI

点击 +1：
  setCount(1) → React 记录新值 → 触发重新渲染
  → 组件函数重新执行
  → useState 返回最新值 count = 1
  → UI 更新为 1
```

## 二、状态类型

### 2.1 基本类型

```tsx
// 数字
const [count, setCount] = useState(0);

// 字符串
const [name, setName] = useState('');

// 布尔值
const [isVisible, setIsVisible] = useState(false);

// null（用于可能没有值的场景）
const [user, setUser] = useState(null);
```

### 2.2 对象状态

```tsx
interface User {
  name: string;
  age: number;
  email: string;
}

const [user, setUser] = useState<User>({
  name: '',
  age: 0,
  email: '',
});

// 更新单个属性（必须展开其他属性）
setUser({ ...user, name: '张三' });

// 更新多个属性
setUser({ ...user, name: '张三', age: 25 });
```

**对象更新的坑：**

```tsx
// 错误 ❌：直接修改原对象
user.name = '张三';
setUser(user);  // React 比较引用相同，不会触发渲染！

// 正确 ✅：创建新对象
setUser({ ...user, name: '张三' });  // 新引用，触发渲染

// 错误 ❌：丢失其他属性
setUser({ name: '张三' });  // age 和 email 丢失了！

// 正确 ✅：展开保留其他属性
setUser({ ...user, name: '张三' });
```

### 2.3 数组状态

```tsx
const [items, setItems] = useState<string[]>([]);

// 添加：展开 + 追加
setItems([...items, '新项目']);

// 删除：filter 过滤
setItems(items.filter(item => item !== '要删的'));

// 修改：map 替换
setItems(items.map(item =>
  item === '旧值' ? '新值' : item
));
```

## 三、函数式更新

当新状态依赖旧状态时，使用函数式更新可以避免"闭包陷阱"。

```tsx
// 问题场景：连续更新
const handleClick = () => {
  setCount(count + 1);  // count = 0, 设置为 1
  setCount(count + 1);  // count 还是 0（闭包捕获），还是设置为 1
  setCount(count + 1);  // count 还是 0，还是设置为 1
  // 结果：count = 1，而不是 3！
};

// 解决方案：函数式更新
const handleClick = () => {
  setCount(prev => prev + 1);  // prev = 0, 设置为 1
  setCount(prev => prev + 1);  // prev = 1, 设置为 2
  setCount(prev => prev + 1);  // prev = 2, 设置为 3
  // 结果：count = 3 ✅
};
```

**何时使用函数式更新：**

| 场景 | 推荐方式 |
|------|----------|
| 新值不依赖旧值 | `setCount(100)` |
| 新值依赖旧值 | `setCount(prev => prev + 1)` |
| 连续多次更新 | `setCount(prev => prev + 1)` |
| 基于旧值的复杂计算 | `setCount(prev => prev * 2 + 1)` |

## 四、不可变更新模式

React 要求状态更新必须"不可变"——即创建新值，而不是修改原值。

### 4.1 数组操作速查表

```tsx
const [todos, setTodos] = useState<Todo[]>([]);

// 添加一项
setTodos([...todos, newTodo]);

// 添加多项
setTodos([...todos, todo1, todo2]);

// 删除一项（按 id）
setTodos(todos.filter(t => t.id !== id));

// 删除一项（按索引）
setTodos(todos.filter((_, i) => i !== index));

// 修改一项
setTodos(todos.map(t =>
  t.id === id ? { ...t, completed: true } : t
));

// 替换一项
setTodos(todos.map(t =>
  t.id === id ? newTodo : t
));

// 排序（先复制再排序）
setTodos([...todos].sort((a, b) => a.name.localeCompare(b.name)));

// 反转
setTodos([...todos].reverse());
```

### 4.2 对象操作速查表

```tsx
const [form, setForm] = useState({ name: '', email: '', age: 0 });

// 更新单个字段
setForm({ ...form, name: '张三' });

// 动态字段名（表单通用）
const handleChange = (e) => {
  setForm({ ...form, [e.target.name]: e.target.value });
};

// 嵌套对象更新
const [user, setUser] = useState({
  name: '张三',
  address: { city: '北京', zip: '100000' },
});
setUser({
  ...user,
  address: { ...user.address, city: '上海' },
});
```

## 五、实战：计数器组件

```tsx
import { useState } from 'react';

const Counter = () => {
  const [count, setCount] = useState(0);

  return (
    <div className="widget">
      <h3>Pattern 1: useState</h3>

      {/* 显示当前值 */}
      <div className="text-4xl font-bold my-4">{count}</div>

      {/* 操作按钮 */}
      <div className="flex gap-2">
        <button
          className="btn btn-primary"
          onClick={() => setCount(prev => prev + 1)}
        >
          +1
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setCount(prev => prev - 1)}
        >
          -1
        </button>
        <button
          className="btn btn-destructive"
          onClick={() => setCount(0)}
        >
          重置
        </button>
      </div>
    </div>
  );
};
```

## 六、常见错误

### 错误 1：忘记使用状态更新函数

```tsx
// 错误 ❌
count = count + 1;  // 直接赋值，React 不知道

// 正确 ✅
setCount(count + 1);
```

### 错误 2：在渲染中调用 setState

```tsx
// 错误 ❌：无限循环
const Component = () => {
  const [count, setCount] = useState(0);
  setCount(count + 1);  // 渲染 → 更新 → 渲染 → 更新...
  return <p>{count}</p>;
};
```

### 错误 3：直接修改对象/数组

```tsx
// 错误 ❌
const [arr, setArr] = useState([1, 2, 3]);
arr.push(4);
setArr(arr);  // 引用相同，React 不会重新渲染

// 正确 ✅
setArr([...arr, 4]);  // 新数组，触发渲染
```

## 学习检查清单

- [ ] 理解 useState 的基本语法
- [ ] 能区分普通变量和状态变量
- [ ] 掌握对象和数组的不可变更新
- [ ] 理解函数式更新的使用场景
- [ ] 知道为什么不能直接修改状态

## 下一篇

[Pattern 2: useEffect - 让组件与外部世界交互](/posts/react-pattern-2-useeffect)

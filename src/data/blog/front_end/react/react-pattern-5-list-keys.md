---
title: React 进阶 Pattern 5：列表渲染与 Key - 高效渲染一组数据
author: Aidenz
pubDatetime: 2026-04-10T12:00:00Z
slug: react-pattern-5-list-keys
featured: false
draft: false
series: React 进阶 Pattern
seriesOrder: 5
tags:
  - React
  - 列表渲染
  - Key
  - 前端
description: 深入理解 React 列表渲染机制，包括 map 方法、key 的作用与选择、不可变更新、进度统计、列表操作等实战场景。
---

## 列表渲染的核心

在 React 中渲染列表，本质上就是把一个**数组**转换成一个**JSX 元素数组**。最常用的方式是 `Array.map()`。

```tsx
const names = ['张三', '李四', '王五'];

const NameList = () => {
  return (
    <ul>
      {names.map(name => (
        <li key={name}>{name}</li>
      ))}
    </ul>
  );
};
```

## 一、key 的作用

### 1.1 为什么需要 key？

React 需要一种方式来识别列表中的每个元素，以便在数据变化时高效地更新 DOM。

```
没有 key 时：
  React 按顺序逐个比较 → 插入/删除时导致大量不必要的 DOM 操作

有 key 时：
  React 通过 key 精确定位 → 只更新变化的元素，效率更高
```

### 1.2 key 的选择

```tsx
// ✅ 最佳：使用唯一 id
{todos.map(todo => <TodoItem key={todo.id} todo={todo} />)}

// ⚠️ 勉强：使用索引（仅在列表完全静态时可用）
{items.map((item, index) => <Item key={index} item={item} />)}

// ❌ 错误：不提供 key
{items.map(item => <Item item={item} />)}
// React 会警告：Each child in a list should have a unique "key" prop
```

### 1.3 为什么不能用索引作为 key？

```tsx
// 场景：在列表头部插入一项

// 用索引作为 key
// 之前: key=0→A, key=1→B, key=2→C
// 之后: key=0→新项, key=1→A, key=2→B, key=3→C
// React 认为：key=0 的内容从 A 变成了新项，key=1 从 B 变成 A...
// 结果：更新了所有元素（低效！）

// 用唯一 id 作为 key
// 之前: key=a→A, key=b→B, key=c→C
// 之后: key=新→新项, key=a→A, key=b→B, key=c→C
// React 认为：只新增了一个元素
// 结果：只插入一个元素（高效！）
```

**何时可以用索引：**

| 场景 | 是否可以用索引 |
|------|----------------|
| 列表不会增删改 | 可以 |
| 列表会排序 | 不可以 |
| 列表会在头部/中间插入 | 不可以 |
| 列表项有唯一 id | 不要用索引，用 id |

## 二、Todo List 实战

### 2.1 数据结构

```tsx
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

const initialTodos: Todo[] = [
  { id: 1, text: '学习 useState', completed: true },
  { id: 2, text: '学习 useEffect', completed: false },
  { id: 3, text: '学习 Props', completed: false },
  { id: 4, text: '学习条件渲染', completed: false },
  { id: 5, text: '学习列表渲染', completed: false },
];
```

### 2.2 完整组件

```tsx
const TodoList = () => {
  const [todos, setTodos] = useState<Todo[]>(initialTodos);

  // 切换完成状态（不可变更新）
  const toggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, completed: !todo.completed }
        : todo
    ));
  };

  // 计算进度
  const completedCount = todos.filter(t => t.completed).length;
  const progress = (completedCount / todos.length) * 100;

  return (
    <div className="widget">
      <h3>Pattern 5: 列表渲染</h3>

      {/* 进度条 */}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p>{completedCount}/{todos.length} 已完成</p>

      {/* 列表 */}
      <div className="todo-list">
        {todos.map(todo => (
          <div
            key={todo.id}
            className={`todo-item ${todo.completed ? 'completed' : ''}`}
            onClick={() => toggleTodo(todo.id)}
          >
            <span>{todo.completed ? '✅' : '⬜'}</span>
            <span>{todo.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

## 三、列表操作模式

### 3.1 添加

```tsx
const addTodo = (text: string) => {
  const newTodo: Todo = {
    id: Date.now(),  // 简单的 id 生成
    text,
    completed: false,
  };
  setTodos([...todos, newTodo]);
};
```

### 3.2 删除

```tsx
const deleteTodo = (id: number) => {
  setTodos(todos.filter(todo => todo.id !== id));
};
```

### 3.3 修改

```tsx
const editTodo = (id: number, newText: string) => {
  setTodos(todos.map(todo =>
    todo.id === id ? { ...todo, text: newText } : todo
  ));
};
```

### 3.4 切换状态

```tsx
const toggleTodo = (id: number) => {
  setTodos(todos.map(todo =>
    todo.id === id
      ? { ...todo, completed: !todo.completed }
      : todo
  ));
};
```

### 3.5 批量操作

```tsx
// 全部标记为已完成
const completeAll = () => {
  setTodos(todos.map(todo => ({ ...todo, completed: true })));
};

// 清除已完成的
const clearCompleted = () => {
  setTodos(todos.filter(todo => !todo.completed));
};

// 反转列表
const reverseList = () => {
  setTodos([...todos].reverse());
};
```

## 四、列表统计

### 4.1 useMemo 优化统计计算

```tsx
import { useMemo } from 'react';

const TodoList = () => {
  const [todos, setTodos] = useState<Todo[]>(initialTodos);

  // 使用 useMemo 缓存计算结果
  const stats = useMemo(() => ({
    total: todos.length,
    completed: todos.filter(t => t.completed).length,
    remaining: todos.filter(t => !t.completed).length,
    progress: todos.length > 0
      ? (todos.filter(t => t.completed).length / todos.length) * 100
      : 0,
  }), [todos]);  // 只在 todos 变化时重新计算

  return (
    <div>
      <p>总计：{stats.total}</p>
      <p>已完成：{stats.completed}</p>
      <p>剩余：{stats.remaining}</p>
      <div className="progress-bar">
        <div style={{ width: `${stats.progress}%` }} />
      </div>
    </div>
  );
};
```

### 4.2 直接计算 vs useMemo

```tsx
// 直接计算（简单场景足够）
const completedCount = todos.filter(t => t.completed).length;
// 每次渲染都执行，但对于小列表性能无影响

// useMemo（复杂计算时使用）
const stats = useMemo(() => {
  // 大量计算...
  return { total, completed, remaining };
}, [todos]);
// 只在 todos 变化时执行，避免不必要的计算
```

**何时用 useMemo：**

| 场景 | 是否需要 useMemo |
|------|-------------------|
| 简单的 filter/map | 不需要 |
| 大数组（>1000 项）的复杂计算 | 推荐 |
| 计算结果作为子组件的 prop | 推荐（避免子组件不必要的重渲染） |
| 计算结果用于 useEffect 的依赖 | 推荐（保持引用稳定） |

## 五、嵌套列表

```tsx
interface Category {
  id: number;
  name: string;
  items: string[];
}

const categories: Category[] = [
  { id: 1, name: '水果', items: ['苹果', '香蕉', '橙子'] },
  { id: 2, name: '蔬菜', items: ['白菜', '萝卜', '土豆'] },
];

const NestedList = () => {
  return (
    <div>
      {categories.map(category => (
        <div key={category.id}>
          <h3>{category.name}</h3>
          <ul>
            {category.items.map((item, index) => (
              <li key={`${category.id}-${index}`}>{item}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
};
```

## 六、常见错误

### 错误 1：忘记 key

```tsx
// 错误 ❌
{todos.map(todo => <TodoItem todo={todo} />)}

// 正确 ✅
{todos.map(todo => <TodoItem key={todo.id} todo={todo} />)}
```

### 错误 2：key 不唯一

```tsx
// 错误 ❌：多个元素可能有相同的 name
{users.map(user => <UserCard key={user.name} user={user} />)}

// 正确 ✅：使用唯一 id
{users.map(user => <UserCard key={user.id} user={user} />)}
```

### 错误 3：在循环外使用索引

```tsx
// 错误 ❌：key 是固定的，不是索引
{todos.map(todo => <TodoItem key="todo" todo={todo} />)}

// 正确 ✅
{todos.map(todo => <TodoItem key={todo.id} todo={todo} />)}
```

### 错误 4：直接修改列表

```tsx
// 错误 ❌
const deleteTodo = (id) => {
  const index = todos.findIndex(t => t.id === id);
  todos.splice(index, 1);  // 直接修改原数组
  setTodos(todos);  // 引用相同，不触发渲染
};

// 正确 ✅
const deleteTodo = (id) => {
  setTodos(todos.filter(t => t.id !== id));  // 创建新数组
};
```

## 学习检查清单

- [ ] 理解 key 的作用和选择原则
- [ ] 掌握 map 渲染列表的写法
- [ ] 掌握添加、删除、修改的不可变更新模式
- [ ] 能使用 useMemo 优化统计计算
- [ ] 避免常见的 key 相关错误

## 下一篇

[Pattern 6: 事件处理与表单 - 处理用户输入和交互](/posts/react/react-pattern-6-forms-events)

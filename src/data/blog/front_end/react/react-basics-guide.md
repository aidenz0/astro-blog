---
title: React 入门完全指南 - 从零开始构建现代 UI
author: Aidenz
pubDatetime: 2026-04-10T07:00:00Z
slug: react-basics-guide
featured: true
draft: false
tags:
  - React
  - 前端
  - 入门教程
description: 基于实际项目，系统介绍 React 核心概念，包括组件、Props、useState、事件处理、表单、条件渲染、列表渲染等基础知识，适合零基础初学者。
---

## 引言

React 是由 Meta（Facebook）开发的前端 UI 库，采用组件化思想构建用户界面。本文基于两个实战项目（基础计数器 + Todo List 应用），系统梳理 React 入门阶段需要掌握的核心知识点。

## 一、项目初始化与工具链

### 1.1 使用 Vite 创建 React 项目

```bash
# 使用 pnpm 创建 Vite + React 项目
pnpm create vite my-react-app --template react

# 如果需要 TypeScript 支持
pnpm create vite my-react-app --template react-ts

# 安装依赖并启动
cd my-react-app
pnpm install
pnpm run dev
```

**Vite vs Webpack 对比：**

| 特性 | Vite | Webpack |
|------|------|---------|
| 启动速度 | 极快（原生 ESM） | 较慢（打包所有模块） |
| 热更新 | 毫秒级 | 秒级 |
| 配置复杂度 | 开箱即用 | 需要大量配置 |
| 生态成熟度 | 快速增长 | 非常成熟 |

### 1.2 项目结构

```
my-react-app/
├── index.html            # 入口 HTML
├── package.json          # 依赖和脚本
├── vite.config.ts        # Vite 配置
├── tsconfig.json         # TypeScript 配置
├── public/               # 静态资源
│   └── favicon.svg
└── src/
    ├── main.tsx          # 应用入口
    ├── App.tsx           # 根组件
    ├── index.css         # 全局样式
    └── components/       # 组件目录
        └── HelloWorld/
            └── index.tsx
```

### 1.3 代码质量工具

```bash
# ESLint - 代码规范检查
pnpm add -D eslint @typescript-eslint/parser eslint-plugin-react-hooks

# Prettier - 代码格式化
pnpm add -D prettier eslint-config-prettier
```

**Prettier 配置示例（`.prettierrc`）：**

```json
{
  "semi": true,
  "singleQuote": true,
  "trailingComma": "all",
  "printWidth": 80,
  "tabWidth": 2
}
```

## 二、React 入口与渲染

### 2.1 应用入口（main.tsx）

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

**关键概念：**

| API | 作用 |
|-----|------|
| `createRoot` | React 18+ 的并发渲染 API，替代旧的 `ReactDOM.render` |
| `StrictMode` | 开发模式下的额外检查，帮助发现潜在问题（不会影响生产环境） |
| `!`（非空断言） | TypeScript 语法，告诉编译器该元素一定存在 |

### 2.2 JSX 语法

JSX 是 JavaScript 的语法扩展，让你在 JS 中编写类似 HTML 的结构。

```tsx
// JSX 本质上是 React.createElement() 的语法糖
const element = <h1 className="title">Hello World</h1>;

// 等价于
const element = React.createElement('h1', { className: 'title' }, 'Hello World');
```

**JSX 注意事项：**

```tsx
// 1. 必须有一个根元素
// 错误 ❌
return (
  <h1>标题</h1>
  <p>段落</p>
);

// 正确 ✅ - 使用 Fragment 包裹
return (
  <>
    <h1>标题</h1>
    <p>段落</p>
  </>
);

// 2. className 而不是 class
<div className="container">...</div>

// 3. 使用 {} 嵌入 JavaScript 表达式
const name = "React";
<h1>Hello {name}!</h1>

// 4. 条件表达式
{isLoggedIn ? <p>欢迎回来</p> : <p>请登录</p>}

// 5. 注释要用 {/* */}
{/* 这是 JSX 注释 */}
```

## 三、组件与 Props

### 3.1 函数组件

组件是 React 的核心概念，每个组件都是一个返回 JSX 的函数。

```tsx
// 箭头函数风格（推荐）
const HelloWorld = () => {
  return <h1>Hello World</h1>;
};

// function 声明风格
function Greeting() {
  return <h1>你好</h1>;
}

// 组件使用
const App = () => {
  return (
    <>
      <HelloWorld />
      <Greeting />
    </>
  );
};
```

### 3.2 Props - 组件间传递数据

Props（属性）是父组件向子组件传递数据的方式。

```tsx
// 定义 Props 类型（TypeScript）
interface HelloWorldProps {
  title: string;
  age: number;
}

// 子组件接收 Props
const HelloWorld = ({ title, age }: HelloWorldProps) => {
  return (
    <div>
      <h1>{title}</h1>
      <p>年龄：{age}</p>
    </div>
  );
};

// 父组件传递 Props
const App = () => {
  return <HelloWorld title="Hello React" age={25} />;
};
```

**Props 特性：**

| 特性 | 说明 |
|------|------|
| 只读 | 子组件不能修改 Props（单向数据流） |
| 任意类型 | 可以传递字符串、数字、数组、对象、函数 |
| 解构赋值 | 推荐在参数中直接解构 |
| 默认值 | 可以设置默认值 `({ name = "默认" })` |

### 3.3 Render Props 模式

Render Props 是一种高级模式，让父组件控制子组件的部分渲染逻辑。

```tsx
interface CounterProps {
  render: (count: number) => React.ReactNode;
}

const Counter = ({ render }: CounterProps) => {
  const [count, setCount] = useState(0);

  return (
    <div>
      <button onClick={() => setCount(count + 1)}>+1</button>
      {render(count)}  {/* 调用父组件传入的渲染函数 */}
    </div>
  );
};

// 父组件通过 render prop 控制显示
const App = () => {
  return (
    <Counter
      render={(count) => <p>当前计数：{count}</p>}
    />
  );
};
```

### 3.4 Callback Props - 子组件通知父组件

```tsx
interface HelloWorldProps {
  title: string;
  onChange: (count: number) => void;  // 回调函数
}

const HelloWorld = ({ title, onChange }: HelloWorldProps) => {
  const [count, setCount] = useState(0);

  const handleIncrement = () => {
    const newCount = count + 1;
    setCount(newCount);
    onChange(newCount);  // 通知父组件
  };

  return (
    <div>
      <h1>{title}</h1>
      <button onClick={handleIncrement}>+1</button>
    </div>
  );
};

// 父组件接收通知
const App = () => {
  const handleChange = (count: number) => {
    console.log('子组件的 count 变为：', count);
  };

  return <HelloWorld title="计数器" onChange={handleChange} />;
};
```

**Props 传递模式对比：**

| 模式 | 方向 | 用途 |
|------|------|------|
| 普通 Props | 父 → 子 | 传递数据 |
| Render Props | 父 → 子 | 父组件控制子组件渲染 |
| Callback Props | 子 → 父 | 子组件通知父组件 |

## 四、useState - 状态管理

### 4.1 基本用法

`useState` 是 React 最核心的 Hook，用于在函数组件中管理状态。

```tsx
import { useState } from 'react';

const Counter = () => {
  // 声明状态：[当前值, 更新函数] = useState(初始值)
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>计数：{count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
      <button onClick={() => setCount(count - 1)}>-1</button>
      <button onClick={() => setCount(0)}>重置</button>
    </div>
  );
};
```

**useState 关键规则：**

```tsx
// 1. 状态更新是异步的
const handleClick = () => {
  setCount(count + 1);
  console.log(count); // 仍然是旧值！
};

// 2. 使用函数式更新（依赖上一个状态时）
setCount(prev => prev + 1);  // 推荐

// 3. 状态更新会触发组件重新渲染
// 调用 setCount → 组件函数重新执行 → UI 更新
```

### 4.2 复杂状态 - 对象和数组

```tsx
// 对象状态
const [user, setUser] = useState({ name: '', age: 0 });

// 更新对象（必须创建新对象）
setUser({ ...user, name: '张三' });       // ✅ 正确
setUser({ ...user, age: 25 });             // ✅ 正确
// user.name = '张三'; setUser(user);      // ❌ 错误：直接修改

// 数组状态
const [items, setItems] = useState<string[]>([]);

// 添加元素
setItems([...items, '新项目']);             // ✅ 展开运算符

// 删除元素
setItems(items.filter(item => item !== '要删的'));  // ✅ filter

// 修改元素
setItems(items.map(item =>
  item === '旧值' ? '新值' : item          // ✅ map
));
```

### 4.3 不可变更新模式大全

```tsx
const [todos, setTodos] = useState([]);

// 添加 - 展开运算符追加
const addTodo = (newTodo) => {
  setTodos([...todos, newTodo]);
};

// 删除 - filter 过滤
const deleteTodo = (id) => {
  setTodos(todos.filter(todo => todo.id !== id));
};

// 修改单个属性 - map + 展开
const toggleComplete = (id) => {
  setTodos(todos.map(todo =>
    todo.id === id
      ? { ...todo, isCompleted: !todo.isCompleted }  // 创建新对象
      : todo                                          // 保持不变
  ));
};

// 修改多个属性
const editTodo = (id, newContent) => {
  setTodos(todos.map(todo =>
    todo.id === id
      ? { ...todo, content: newContent, isEditing: false }
      : todo
  ));
};
```

**为什么不能直接修改状态？**

```tsx
// 错误方式 ❌
const toggleWrong = (id) => {
  const todo = todos.find(t => t.id === id);
  todo.isCompleted = !todo.isCompleted;  // 直接修改了原对象
  setTodos(todos);  // React 比较引用相同，认为没变化，不重新渲染！
};

// 正确方式 ✅
const toggleRight = (id) => {
  setTodos(todos.map(todo =>
    todo.id === id
      ? { ...todo, isCompleted: !todo.isCompleted }  // 创建新对象，引用不同
      : todo
  ));
};
```

## 五、事件处理

### 5.1 事件绑定

```tsx
const Button = () => {
  // 事件处理函数
  const handleClick = () => {
    console.log('按钮被点击');
  };

  return (
    // 方式1：内联函数（简单场景）
    <button onClick={() => console.log('点击')}>点击</button>

    // 方式2：引用函数（推荐）
    <button onClick={handleClick}>点击</button>

    // 注意：不要加括号！
    // <button onClick={handleClick()}>  ❌ 会立即执行
  );
};
```

### 5.2 事件对象

```tsx
const Form = () => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();  // 阻止表单默认提交行为
    console.log('表单提交');
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);  // 获取输入值
  };

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log(e.target);        // 触发事件的元素
    console.log(e.clientX);       // 鼠标位置
  };

  return (
    <form onSubmit={handleSubmit}>
      <input onChange={handleChange} />
      <button onClick={handleClick}>提交</button>
    </form>
  );
};
```

### 5.3 常用事件类型

```tsx
// 鼠标事件
<div
  onClick={() => console.log('点击')}
  onMouseEnter={() => console.log('鼠标进入')}
  onMouseLeave={() => console.log('鼠标离开')}
/>

// 键盘事件
<input
  onKeyDown={(e) => {
    if (e.key === 'Enter') console.log('按下回车');
  }}
/>

// 表单事件
<input
  onFocus={() => console.log('获得焦点')}
  onBlur={() => console.log('失去焦点')}
  onChange={(e) => console.log('值变化:', e.target.value)}
/>
```

## 六、表单处理

### 6.1 受控组件

受控组件的值由 React 状态控制，是 React 处理表单的标准方式。

```tsx
const CreateForm = ({ addTodo }) => {
  // 用状态控制输入值
  const [content, setContent] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();      // 阻止页面刷新

    if (!content.trim()) return;  // 校验：不能为空

    addTodo(content);        // 调用父组件方法
    setContent('');          // 清空输入框
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={content}                        // 值来自状态
        onChange={(e) => setContent(e.target.value)}  // 更新状态
        placeholder="输入待办事项..."
      />
      <button type="submit">添加</button>
    </form>
  );
};
```

**受控 vs 非受控组件：**

| 特性 | 受控组件 | 非受控组件 |
|------|----------|------------|
| 值的来源 | React 状态 | DOM 自身 |
| 获取值 | `onChange` + 状态 | `useRef` |
| 实时校验 | 容易 | 需要额外处理 |
| 推荐场景 | 大多数情况 | 文件上传等 |

### 6.2 编辑表单

```tsx
const EditForm = ({ todo, editTodo }) => {
  // 用已有内容初始化状态
  const [content, setContent] = useState(todo.content);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    editTodo(todo.id, content);  // 传递 id 和新内容
  };

  return (
    <form onSubmit={handleSubmit} className="edit-form">
      <input
        type="text"
        value={content}
        onChange={(e) => setContent(e.target.value)}
      />
      <button type="submit">保存</button>
    </form>
  );
};
```

## 七、条件渲染

### 7.1 三元运算符

```tsx
const Todo = ({ todo }) => {
  // 如果正在编辑，显示编辑表单；否则显示普通视图
  return (
    <div>
      {todo.isEditing ? (
        <EditForm todo={todo} />
      ) : (
        <p>{todo.content}</p>
      )}
    </div>
  );
};
```

### 7.2 逻辑与（&&）

```tsx
// 只在条件为 true 时渲染
const Notification = ({ message }) => {
  return (
    <div>
      {message && <p className="alert">{message}</p>}
    </div>
  );
};
```

### 7.3 动态 className

```tsx
const Todo = ({ todo }) => {
  // 根据状态动态拼接 className
  const className = `todo ${todo.isCompleted ? 'completed' : ''}`;

  return (
    <div className={className}>
      <p>{todo.content}</p>
    </div>
  );
};
```

```css
/* 对应的 CSS */
.completed {
  text-decoration: line-through;
  opacity: 0.6;
}
```

## 八、列表渲染

### 8.1 使用 map 渲染列表

```tsx
const TodoList = ({ todos }) => {
  return (
    <div>
      {todos.map(todo => (
        <Todo
          key={todo.id}    {/* key 是必须的，用于 React 识别元素 */}
          todo={todo}
        />
      ))}
    </div>
  );
};
```

### 8.2 key 的重要性

```tsx
// key 帮助 React 识别哪些元素变化了

// ✅ 正确：使用唯一 id
{todos.map(todo => <Todo key={todo.id} todo={todo} />)}

// ⚠️ 勉强：使用索引（仅在列表不会增删改时可用）
{todos.map((todo, index) => <Todo key={index} todo={todo} />)}

// ❌ 错误：不提供 key（React 会警告）
{todos.map(todo => <Todo todo={todo} />)}
```

**为什么需要 key？**

```
没有 key 时：
React 按顺序逐个比较 → 插入/删除时会导致不必要的 DOM 更新

有 key 时：
React 通过 key 精确定位 → 只更新变化的元素，效率更高
```

## 九、组件组合与数据流

### 9.1 组件层次结构

```
App
└── TodoWrapper          ← 状态提升到这一层
    ├── CreateForm       ← 接收 addTodo 回调
    └── Todo             ← 接收数据和操作函数
        └── EditForm     ← 接收 editTodo 回调
```

### 9.2 Props Drilling - 逐层传递

```tsx
// 状态定义在 TodoWrapper 中
const TodoWrapper = () => {
  const [todos, setTodos] = useState([...]);

  const deleteTodo = (id) => { ... };
  const toggleComplete = (id) => { ... };
  const editTodo = (id, content) => { ... };

  return (
    <div>
      <CreateForm addTodo={addTodo} />
      {todos.map(todo => (
        <Todo
          key={todo.id}
          todo={todo}
          deleteTodo={deleteTodo}       {/* 传递操作函数 */}
          toggleComplete={toggleComplete}
          editTodo={editTodo}
        />
      ))}
    </div>
  );
};

// Todo 组件接收并使用
const Todo = ({ todo, deleteTodo, toggleComplete, editTodo }) => {
  return (
    <div>
      <p onClick={() => toggleComplete(todo.id)}>{todo.content}</p>
      <MdEdit onClick={() => /* 切换编辑状态 */} />
      <MdDeleteSweep onClick={() => deleteTodo(todo.id)} />
    </div>
  );
};
```

**数据流方向：**

```
父组件 ──Props──→ 子组件（数据向下）
子组件 ──Callback──→ 父组件（事件向上）

这是 React 的单向数据流原则
```

## 十、样式处理

### 10.1 CSS 文件

```css
/* App.css */
.todo-wrapper {
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
}

.todo {
  display: flex;
  align-items: center;
  padding: 10px;
  border-bottom: 1px solid #eee;
}

.completed {
  text-decoration: line-through;
  opacity: 0.6;
}
```

```tsx
// 在组件中引入
import './App.css';
```

### 10.2 内联样式

```tsx
// 使用对象语法（注意是双花括号）
<div style={{ cursor: 'pointer', color: 'red' }}>
  内联样式
</div>

// 属性名用驼峰命名
<div style={{ backgroundColor: '#f0f0f0', fontSize: '16px' }}>
  背景色和字号
</div>
```

### 10.3 动态样式

```tsx
const Todo = ({ todo }) => {
  return (
    <div
      className={`todo ${todo.isCompleted ? 'completed' : ''}`}
      style={{
        cursor: 'pointer',
        opacity: todo.isCompleted ? 0.5 : 1,
      }}
    >
      {todo.content}
    </div>
  );
};
```

## 十一、第三方库集成

### 11.1 安装和使用 react-icons

```bash
pnpm add react-icons
```

```tsx
// 按需引入（减少打包体积）
import { MdDeleteSweep, MdEdit } from 'react-icons/md';
import { FaCheck, FaTimes } from 'react-icons/fa';

const Todo = ({ todo, deleteTodo }) => {
  return (
    <div className="todo">
      <p>{todo.content}</p>
      <MdEdit
        style={{ cursor: 'pointer' }}
        onClick={() => /* 编辑 */}
      />
      <MdDeleteSweep
        style={{ cursor: 'pointer' }}
        onClick={() => deleteTodo(todo.id)}
      />
    </div>
  );
};
```

## 十二、完整实战：Todo List 架构拆解

### 12.1 数据结构设计

```tsx
interface Todo {
  id: number;          // 唯一标识
  content: string;     // 待办内容
  isCompleted: boolean; // 是否完成
  isEditing: boolean;  // 是否处于编辑状态
}

// 初始数据
const initialTodos: Todo[] = [
  { id: 1, content: '学习 React 基础', isCompleted: false, isEditing: false },
  { id: 2, content: '完成 Todo List 项目', isCompleted: false, isEditing: false },
];
```

### 12.2 状态操作函数

```tsx
const TodoWrapper = () => {
  const [todos, setTodos] = useState(initialTodos);

  // 添加
  const addTodo = (content: string) => {
    const newTodo = {
      id: Date.now(),       // 简单的 id 生成方式
      content,
      isCompleted: false,
      isEditing: false,
    };
    setTodos([...todos, newTodo]);
  };

  // 删除
  const deleteTodo = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  // 切换完成状态
  const toggleComplete = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, isCompleted: !todo.isCompleted }
        : todo
    ));
  };

  // 切换编辑状态
  const toggleEdit = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, isEditing: !todo.isEditing }
        : todo
    ));
  };

  // 保存编辑
  const editTodo = (id: number, newContent: string) => {
    setTodos(todos.map(todo =>
      todo.id === id
        ? { ...todo, content: newContent, isEditing: false }
        : todo
    ));
  };

  // ...渲染 JSX
};
```

### 12.3 组件通信总结

```
TodoWrapper（状态中心）
│
│  todos, addTodo
├──→ CreateForm（输入新待办）
│
│  todo, deleteTodo, toggleComplete, toggleEdit, editTodo
└──→ Todo（显示单个待办）
     │
     │  todo, editTodo
     └──→ EditForm（编辑待办内容）
```

## 十三、学习检查清单

掌握以下知识点表示你已打好 React 基础：

- [ ] 使用 Vite 创建 React 项目
- [ ] 理解 JSX 语法规则
- [ ] 创建函数组件
- [ ] 使用 Props 传递数据（字符串、数字、函数）
- [ ] 使用 TypeScript 接口定义 Props 类型
- [ ] 理解 Render Props 和 Callback Props 模式
- [ ] 使用 useState 管理基本状态
- [ ] 使用 useState 管理对象和数组状态
- [ ] 掌握不可变更新模式（展开运算符、map、filter）
- [ ] 处理点击、键盘等事件
- [ ] 使用受控组件处理表单
- [ ] 使用三元运算符和 && 进行条件渲染
- [ ] 使用 map 渲染列表并正确使用 key
- [ ] 理解单向数据流和组件通信
- [ ] 使用 CSS 文件和内联样式
- [ ] 集成第三方库（react-icons）

## 总结

React 的核心思想是**组件化**和**单向数据流**。通过本文的学习，你应该掌握了：

1. **组件**：将 UI 拆分为独立、可复用的函数组件
2. **Props**：父组件向子组件传递数据和回调函数
3. **useState**：在组件内管理状态，使用不可变方式更新
4. **事件处理**：响应用户交互，处理表单输入
5. **条件渲染与列表渲染**：根据状态动态展示 UI
6. **组件组合**：通过 Props Drilling 实现多层组件通信

这些是 React 最基础也最核心的知识。下一步可以学习 `useEffect`（副作用处理）、`useContext`（跨层传递状态）、React Router（路由管理）等进阶内容。

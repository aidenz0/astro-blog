---
title: "React 进阶 Pattern 4：条件渲染 - 根据状态显示不同 UI"
author: Aidenz
pubDatetime: 2026-05-03T13:00:00Z
slug: react-pattern-4-conditional-rendering
featured: false
draft: false
tags:
  - React
  - 条件渲染
  - 前端
description: 深入理解 React 条件渲染的多种模式，包括逻辑与、三元运算符、早返回、状态机模式，配合用户资料卡片的加载/错误/成功/空状态实战。
---

## 为什么需要条件渲染？

实际应用中，UI 不是静态的。同一个组件可能需要展示：

- 加载中的 spinner
- 请求失败的错误提示
- 请求成功的数据
- 数据为空时的占位图

这就是**条件渲染**——根据不同的状态，渲染不同的 UI。

## 一、四种基本模式

### 1.1 逻辑与（&&）

当条件为 true 时才渲染某个元素。

```tsx
const Notification = ({ message }: { message: string }) => {
  return (
    <div>
      {message && <p className="alert">{message}</p>}
    </div>
  );
};
```

**注意陷阱：**

```tsx
// 问题 ❌：当 count 为 0 时会渲染 "0" 而不是不渲染
const count = 0;
{count && <p>有 {count} 条消息</p>}  // 渲染 "0"！

// 正确 ✅：转换为布尔值
{!!count && <p>有 {count} 条消息</p>}  // 不渲染
```

### 1.2 三元运算符

二选一的场景。

```tsx
const Greeting = ({ isLoggedIn }: { isLoggedIn: boolean }) => {
  return (
    <div>
      {isLoggedIn ? <p>欢迎回来！</p> : <p>请登录</p>}
    </div>
  );
};
```

### 1.3 早返回（Early Return）

根据状态提前返回不同的 UI，避免深层嵌套。

```tsx
const UserProfile = ({ user, loading, error }: Props) => {
  // 早返回：每种状态直接返回对应的 UI
  if (loading) return <div className="loading">加载中...</div>;
  if (error) return <div className="error">错误：{error}</div>;
  if (!user) return <div className="empty">暂无数据</div>;

  // 主路径：所有条件都不满足时渲染正常 UI
  return (
    <div>
      <h2>{user.name}</h2>
      <p>年龄：{user.age}</p>
    </div>
  );
};
```

**早返回 vs 嵌套条件：**

```tsx
// 差 ❌：深层嵌套，难以阅读
if (!loading) {
  if (!error) {
    if (user) {
      return <div>{user.name}</div>;
    } else {
      return <div>暂无数据</div>;
    }
  } else {
    return <div>错误：{error}</div>;
  }
} else {
  return <div>加载中...</div>;
}

// 好 ✅：早返回，扁平结构
if (loading) return <div>加载中...</div>;
if (error) return <div>错误：{error}</div>;
if (!user) return <div>暂无数据</div>;
return <div>{user.name}</div>;
```

### 1.4 变量存储 JSX

复杂条件时，先计算 JSX 再渲染。

```tsx
const TodoItem = ({ todo }: { todo: Todo }) => {
  let content;

  if (todo.isEditing) {
    content = <EditForm todo={todo} />;
  } else if (todo.completed) {
    content = <s>{todo.text}</s>;  // 删除线
  } else {
    content = <span>{todo.text}</span>;
  }

  return <div className="todo-item">{content}</div>;
};
```

## 二、状态机模式

当一个组件有多个互斥状态时，推荐使用状态机模式。

### 2.1 基本状态机

```tsx
interface User {
  name: string;
  age: number;
}

const UserProfile = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        setLoading(true);
        await new Promise(resolve => setTimeout(resolve, 2000));

        // 随机成功或失败（30% 概率失败）
        if (Math.random() > 0.7) {
          throw new Error('网络请求失败');
        }

        setUser({ name: '张三', age: 25 });
      } catch (err) {
        setError(err instanceof Error ? err.message : '未知错误');
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, []);

  // 渲染逻辑
  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} onRetry={fetchUser} />;
  if (!user) return <EmptyState />;
  return <UserCard user={user} />;
};
```

### 2.2 完整的用户资料组件

```tsx
const UserProfile = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchUser = async () => {
    try {
      setLoading(true);
      setError(null);
      await new Promise(resolve => setTimeout(resolve, 2000));

      if (Math.random() > 0.7) {
        throw new Error('模拟的网络错误');
      }

      setUser({ name: '张三', age: 25 });
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUser();
  }, []);

  return (
    <div className="widget">
      <h3>Pattern 4: 条件渲染</h3>

      {/* 状态1：加载中 */}
      {loading && (
        <div className="status-loading">
          <div className="spinner" />
          <p>加载用户数据中...</p>
        </div>
      )}

      {/* 状态2：错误 */}
      {error && (
        <div className="status-error">
          <p>错误：{error}</p>
          <button onClick={fetchUser}>重试</button>
        </div>
      )}

      {/* 状态3：成功 */}
      {!loading && !error && user && (
        <div className="status-success">
          <h4>{user.name}</h4>
          <p>年龄：{user.age}</p>
          <button onClick={fetchUser}>刷新</button>
        </div>
      )}

      {/* 状态4：空数据 */}
      {!loading && !error && !user && (
        <div className="empty-state">
          <p>暂无用户数据</p>
          <button onClick={fetchUser}>获取数据</button>
        </div>
      )}
    </div>
  );
};
```

## 三、动态 className

### 3.1 模板字符串拼接

```tsx
const TodoItem = ({ todo }: { todo: Todo }) => {
  return (
    <div className={`todo-item ${todo.completed ? 'completed' : ''}`}>
      {todo.text}
    </div>
  );
};
```

### 3.2 多条件拼接

```tsx
const TodoItem = ({ todo }: { todo: Todo }) => {
  const className = [
    'todo-item',
    todo.completed && 'completed',
    todo.isEditing && 'editing',
    todo.priority === 'high' && 'high-priority',
  ]
    .filter(Boolean)  // 过滤掉 false/undefined/null
    .join(' ');

  return <div className={className}>{todo.text}</div>;
};
```

### 3.3 对象映射

```tsx
const Button = ({ variant, size }: ButtonProps) => {
  const className = `btn ${
    {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      destructive: 'btn-destructive',
    }[variant]
  }`;

  return <button className={className}>按钮</button>;
};
```

## 四、条件渲染最佳实践

### 4.1 避免深层嵌套

```tsx
// 差 ❌
{user ? (
  user.posts ? (
    user.posts.length > 0 ? (
      <PostList posts={user.posts} />
    ) : (
      <p>暂无文章</p>
    )
  ) : (
    <p>加载中</p>
  )
) : (
  <p>请登录</p>
)}

// 好 ✅：早返回 + 独立组件
if (!user) return <p>请登录</p>;
if (!user.posts) return <p>加载中</p>;
if (user.posts.length === 0) return <p>暂无文章</p>;
return <PostList posts={user.posts} />;
```

### 4.2 提取独立组件

```tsx
// 当条件渲染逻辑复杂时，提取为独立组件
const UserStatus = ({ user, loading, error }: Props) => {
  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!user) return <EmptyState />;
  return <UserData user={user} />;
};

// 每个状态组件专注于自己的渲染
const LoadingState = () => <div>加载中...</div>;
const ErrorState = ({ error }: { error: string }) => <div>错误：{error}</div>;
const EmptyState = () => <div>暂无数据</div>;
const UserData = ({ user }: { user: User }) => <div>{user.name}</div>;
```

## 五、常见错误

### 错误 1：0 被渲染出来

```tsx
// 错误 ❌
{items.length && <p>共 {items.length} 条</p>}
// items 为空数组时渲染 "0" 而不是不渲染

// 正确 ✅
{items.length > 0 && <p>共 {items.length} 条</p>}
{!!items.length && <p>共 {items.length} 条</p>}
```

### 错误 2：空字符串被渲染

```tsx
// 错误 ❌
{error && <p>{error}</p>}
// error 为 "" 时渲染空 <p></p>

// 正确 ✅
{error && <p>{error}</p>}  // 其实空字符串不会渲染，但 null/undefined 会
// 但要注意：{0 && ...} 和 {false && ...} 的行为
```

### 错误 3：三元运算符嵌套过深

```tsx
// 差 ❌：难以阅读
return a ? b ? c ? <A /> : <B /> : <C /> : <D /> : <E />;

// 好 ✅：使用早返回或变量
if (a && b && c) return <A />;
if (a && b) return <B />;
// ...
```

## 学习检查清单

- [ ] 掌握四种条件渲染模式
- [ ] 能处理加载/错误/成功/空四种状态
- [ ] 理解动态 className 的写法
- [ ] 避免 0 和空字符串被渲染的陷阱
- [ ] 能提取独立的状态组件

## 下一篇

[Pattern 5: 列表渲染与 Key - 高效渲染一组数据](/posts/react/react-pattern-5-list-keys)

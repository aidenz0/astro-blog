---
title: "React 进阶 Pattern 6：事件处理与表单 - 处理用户输入"
author: Aidenz
pubDatetime: 2026-05-03T15:00:00Z
slug: react-pattern-6-forms-events
featured: false
draft: false
tags:
  - React
  - 表单
  - 事件处理
  - 前端
description: 深入理解 React 事件处理与表单模式，包括受控组件、通用 handleChange、表单校验、useCallback 优化、提交历史管理等完整实战。
---

## React 事件处理基础

### 1.1 绑定事件

```tsx
const Button = () => {
  const handleClick = () => {
    console.log('按钮被点击');
  };

  return (
    // ✅ 传递函数引用
    <button onClick={handleClick}>点击</button>

    // ✅ 内联函数（简单场景）
    <button onClick={() => console.log('点击')}>点击</button>

    // ❌ 不要加括号（会立即执行）
    // <button onClick={handleClick()}>点击</button>
  );
};
```

### 1.2 事件对象

```tsx
const Input = () => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);   // 输入的值
    console.log(e.target.name);    // input 的 name 属性
    console.log(e.target.type);    // input 的 type 属性
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();  // 阻止表单默认提交（页面刷新）
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="username" onChange={handleChange} />
    </form>
  );
};
```

### 1.3 常用事件类型

| 事件 | 触发时机 | 常见用途 |
|------|----------|----------|
| `onClick` | 元素被点击 | 按钮、链接 |
| `onChange` | 输入值变化 | 输入框、下拉框 |
| `onSubmit` | 表单提交 | 表单处理 |
| `onFocus` | 元素获得焦点 | 高亮、提示 |
| `onBlur` | 元素失去焦点 | 校验、保存 |
| `onKeyDown` | 键盘按下 | 快捷键、回车提交 |
| `onMouseEnter` | 鼠标进入 | 悬停效果 |
| `onMouseLeave` | 鼠标离开 | 取消悬停 |

## 二、受控组件

### 2.1 基本受控组件

React 控制表单元素的值，通过状态驱动。

```tsx
const SimpleForm = () => {
  const [name, setName] = useState('');

  return (
    <form>
      <input
        type="text"
        value={name}                          // 值来自状态
        onChange={(e) => setName(e.target.value)}  // 变化时更新状态
        placeholder="请输入姓名"
      />
      <p>你输入了：{name}</p>
    </form>
  );
};
```

### 2.2 通用 handleChange

多个表单字段时，用一个通用的 `handleChange` 函数处理所有字段。

```tsx
interface FormData {
  name: string;
  email: string;
  message: string;
}

const ContactForm = () => {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    message: '',
  });

  // 通用的 change 处理函数
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,  // 计算属性名：动态设置字段
    }));
  };

  return (
    <form>
      <input name="name" value={formData.name} onChange={handleChange} />
      <input name="email" value={formData.email} onChange={handleChange} />
      <textarea name="message" value={formData.message} onChange={handleChange} />
    </form>
  );
};
```

**计算属性名 `[name]: value` 的原理：**

```tsx
const name = 'email';
const value = 'test@example.com';

{ [name]: value }
// 等价于
{ email: 'test@example.com' }
```

## 三、表单校验

### 3.1 校验逻辑

```tsx
interface FormErrors {
  name?: string;
  email?: string;
  message?: string;
}

const validate = (formData: FormData): FormErrors => {
  const errors: FormErrors = {};

  if (!formData.name.trim()) {
    errors.name = '姓名不能为空';
  }

  if (!formData.email.trim()) {
    errors.email = '邮箱不能为空';
  } else if (!formData.email.includes('@')) {
    errors.email = '邮箱格式不正确';
  }

  if (!formData.message.trim()) {
    errors.message = '留言不能为空';
  }

  return errors;
};
```

### 3.2 实时校验 vs 提交时校验

```tsx
const ContactForm = () => {
  const [formData, setFormData] = useState<FormData>({...});
  const [errors, setErrors] = useState<FormErrors>({});

  // 提交时校验
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationErrors = validate(formData);

    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);  // 显示错误
      return;  // 阻止提交
    }

    // 校验通过，提交数据
    setErrors({});
    // ... 提交逻辑
  };

  // 失焦时校验（可选）
  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const fieldErrors = validate({ ...formData, [name]: value });
    setErrors(prev => ({ ...prev, [name]: fieldErrors[name] }));
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <input
          name="name"
          value={formData.name}
          onChange={handleChange}
          onBlur={handleBlur}
        />
        {errors.name && <span className="error">{errors.name}</span>}
      </div>
      {/* ... 其他字段 */}
    </form>
  );
};
```

## 四、useCallback 优化

### 4.1 为什么需要 useCallback

```tsx
// 问题：每次渲染都创建新的 handleChange 函数
const ContactForm = () => {
  const [formData, setFormData] = useState({...});
  const [errors, setErrors] = useState({});

  // 每次渲染都创建新函数
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    // 清除该字段的错误
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  };

  return <input onChange={handleChange} />;
};
```

### 4.2 使用 useCallback

```tsx
import { useCallback } from 'react';

const ContactForm = () => {
  const [formData, setFormData] = useState({...});
  const [errors, setErrors] = useState<FormErrors>({});

  // 使用 useCallback 缓存函数
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const { name, value } = e.target;
      setFormData(prev => ({ ...prev, [name]: value }));
      if (errors[name as keyof FormErrors]) {
        setErrors(prev => ({ ...prev, [name]: undefined }));
      }
    },
    [errors]  // 依赖项：errors 变化时才创建新函数
  );

  return <input onChange={handleChange} />;
};
```

### 4.3 useCallback 的使用场景

| 场景 | 是否需要 useCallback |
|------|----------------------|
| 传递给子组件的回调函数 | 推荐（配合 React.memo） |
| 作为 useEffect 的依赖 | 推荐（保持引用稳定） |
| 只在当前组件使用 | 不需要 |
| 复杂的事件处理逻辑 | 看情况 |

## 五、提交与历史管理

### 5.1 完整的表单提交流程

```tsx
interface SubmittedData {
  id: number;
  name: string;
  email: string;
  message: string;
  submittedAt: Date;
}

const ContactForm = () => {
  const [formData, setFormData] = useState<FormData>({
    name: '', email: '', message: '',
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [submittedList, setSubmittedList] = useState<SubmittedData[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [nextId, setNextId] = useState(1);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      // 校验
      const validationErrors = validate(formData);
      if (Object.keys(validationErrors).length > 0) {
        setErrors(validationErrors);
        return;
      }

      // 模拟 API 请求
      setIsSubmitting(true);
      await new Promise(resolve => setTimeout(resolve, 1500));

      // 保存提交数据
      const newSubmission: SubmittedData = {
        id: nextId,
        ...formData,
        submittedAt: new Date(),
      };

      setSubmittedList(prev => [newSubmission, ...prev]);
      setNextId(prev => prev + 1);
      setFormData({ name: '', email: '', message: '' });  // 清空表单
      setErrors({});
      setIsSubmitting(false);
    },
    [formData, nextId]
  );

  return (
    <form onSubmit={handleSubmit}>
      {/* 表单字段 */}
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? '提交中...' : '提交'}
      </button>
    </form>
  );
};
```

### 5.2 提交历史与统计

```tsx
const submissionStats = useMemo(() => ({
  total: submittedList.length,
  uniqueEmails: new Set(submittedList.map(s => s.email)).size,
  avgMessageLength: submittedList.length > 0
    ? Math.round(
        submittedList.reduce((sum, s) => sum + s.message.length, 0)
        / submittedList.length
      )
    : 0,
}), [submittedList]);
```

### 5.3 删除操作

```tsx
// 删除单条
const handleDelete = useCallback((id: number) => {
  setSubmittedList(prev => prev.filter(item => item.id !== id));
}, []);

// 清空全部
const handleDeleteAll = useCallback(() => {
  setSubmittedList([]);
}, []);
```

## 六、键盘事件处理

```tsx
const InputWithEnter = () => {
  const [value, setValue] = useState('');

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      // 执行提交逻辑
      console.log('提交：', value);
      setValue('');
    }
  };

  return (
    <input
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyPress={handleKeyPress}
      placeholder="按回车提交"
    />
  );
};
```

## 七、常见错误

### 错误 1：忘记 preventDefault

```tsx
// 错误 ❌：表单提交会刷新页面
const handleSubmit = (e: React.FormEvent) => {
  // 没有 e.preventDefault()
  // 页面刷新，状态丢失
};

// 正确 ✅
const handleSubmit = (e: React.FormEvent) => {
  e.preventDefault();
  // 提交逻辑
};
```

### 错误 2：onChange 中直接修改状态

```tsx
// 错误 ❌
const handleChange = (e) => {
  formData.name = e.target.value;  // 直接修改
  setFormData(formData);  // 引用相同，不触发渲染
};

// 正确 ✅
const handleChange = (e) => {
  setFormData({ ...formData, [e.target.name]: e.target.value });
};
```

### 错误 3：在渲染中发送请求

```tsx
// 错误 ❌
const Component = () => {
  fetch('/api/submit', { method: 'POST', body: data });  // 每次渲染都发
  return <div>...</div>;
};

// 正确 ✅：在事件处理函数或 useEffect 中
const handleSubmit = () => {
  fetch('/api/submit', { method: 'POST', body: data });
};
```

## 学习检查清单

- [ ] 理解 React 事件处理的语法
- [ ] 掌握受控组件的写法
- [ ] 能使用通用 handleChange 处理多字段
- [ ] 理解表单校验的实现方式
- [ ] 掌握 useCallback 的使用场景
- [ ] 能实现完整的表单提交流程

## 下一篇

[Pattern 7: Context API - 跨组件共享状态](/posts/react/react-pattern-7-context-api)

---
title: Python 异步编程完全指南
author: Aidenz
pubDatetime: 2026-05-01T05:00:00Z
slug: python-async-programming
featured: false
draft: false
tags:
  - Python
  - 异步编程
  - asyncio
  - 并发
description: 深入理解 Python 异步编程的核心概念，从事件循环到 async/await 语法，掌握高并发编程的关键技术。
---

## 引言

在现代 Python 开发中，随着应用对高并发、高性能的需求不断增加，传统同步编程方式在处理大量 I/O 操作时逐渐显现出局限性。异步编程通过提升程序的并发能力和资源利用率，已成为 Web 开发、微服务、实时通讯、数据抓取等领域的核心技术。

## 一、同步编程的困境

在深入异步编程之前，我们先来理解传统同步编程在处理 **I/O 密集型任务** 时面临的问题。

### 1.1 I/O 密集型任务的特点

I/O 密集型任务的特点是程序在执行过程中需要频繁等待 **输入/输出操作完成**，常见场景包括：

- 等待网络请求返回数据
- 等待磁盘读写完成
- 等待数据库查询结果

### 1.2 同步代码的问题

在等待期间，**程序会阻塞**，**CPU 处于空闲状态**，无法继续执行其他任务，从而大大降低整体效率。

```python
import time

def task1():
    time.sleep(5)  # 模拟一个耗时 5 秒的 I/O 操作
    return 10

def task2():
    time.sleep(3)  # 模拟一个耗时 3 秒的 I/O 操作
    return 20

def main():
    result = task1()
    print('任务1执行结果:', result)
    result = task2()
    print('任务2执行结果:', result)

if __name__ == '__main__':
    start = time.time()
    main()
    print('总耗时:', time.time() - start)
```

**执行结果：**
```
任务1执行结果: 10
任务2执行结果: 20
总耗时: 8.023201704025269
```

可以看到，`task1()` 执行时必须等待 5 秒，`task2()` 只有在 `task1()` 完成后才能开始，整体耗时约 **8 秒**。

### 1.3 理想情况

当 task1 处于等待状态时，CPU 应该能够切换去执行其他任务（比如 task2），而不是空等。这正是异步编程要解决的核心问题。

## 二、异步编程解决方案

Python 内置的 `asyncio` 模块为我们提供了一套完整的异步编程机制 —— **事件循环（Event Loop）**。

### 2.1 事件循环工作原理

事件循环的工作流程如下：

1. 创建一个事件循环
2. 将需要执行的任务注册到事件循环中
3. 启动事件循环，开始调度和执行各个任务

### 2.2 两个关键规则

为了让事件循环正确地识别和调度任务，需要遵循以下规则：

- **`async def`**：定义异步函数，表示这是一个需要交给事件循环管理的协程函数
- **`await`**：在任务函数内部，当遇到需要等待的异步操作时使用，告诉事件循环此处可以挂起当前任务

### 2.3 异步代码示例

```python
import asyncio
import time

async def task1():
    print('task1 开始执行')
    await asyncio.sleep(5)
    print('task1 结束执行')
    return 10

async def task2():
    print('task2 开始执行')
    await asyncio.sleep(3)
    print('task2 结束执行')
    return 20

async def main():
    print('main 开始执行')
    event_loop = asyncio.get_running_loop()
    t1 = event_loop.create_task(task1())
    t2 = event_loop.create_task(task2())

    result = await t1
    print('任务1执行结果:', result)

    result = await t2
    print('任务2执行结果:', result)

    print('main 结束执行')

if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    print('总耗时:', time.time() - start)
```

**执行结果：**
```
main 开始执行
task1 开始执行
task2 开始执行
task2 结束执行
task1 结束执行
任务1执行结果: 10
任务2执行结果: 20
main 结束执行
总耗时: 5.013811826705933
```

通过异步改造，两个任务的执行时间从 **8 秒缩减到 5 秒**。

### 2.4 简化写法

使用 `asyncio.gather()` 可以更简洁地处理多个并发任务：

```python
async def main():
    results = await asyncio.gather(task1(), task2())
    print(results)

if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    print('总耗时:', time.time() - start)
```

## 三、核心概念详解

### 3.1 async 关键字

`async` 关键字用于定义异步函数（协程函数），这种函数可以执行非阻塞操作。

```python
import asyncio

async def task():
    print('task')

def demo():
    coro = task()
    print(type(coro))  # <class 'coroutine'>

if __name__ == '__main__':
    demo()
```

**重要理解：**

- 协程函数的返回值不是直接结果，而是一个**协程对象**
- 协程对象是一个未开始执行的任务，必须通过事件循环来调度

```python
async def task():
    print('task')

def demo():
    coro = task()
    asyncio.run(coro)  # 将协程对象注册到事件循环中执行

if __name__ == '__main__':
    demo()
```

### 3.2 await 关键字

`await` 关键字用在 `async def` 定义的协程函数中，用于暂停协程的执行，直到异步操作完成。

```python
import asyncio

async def sub_task():
    print('sub task')
    return 100

async def task():
    result = await sub_task()  # 暂停当前任务，等待 sub_task 完成
    print('result:', result)

def demo():
    coro = task()
    asyncio.run(coro)

if __name__ == '__main__':
    demo()
```

**await 的使用规则：**

- `await` 后面可以跟 `coroutine`、`future`、`task` 对象
- 如果 `await` 的是其他类型对象会报错：`TypeError: xxx can't be used in 'await' expression`

### 3.3 任务切换时机

当 `await` 后面是协程对象时，事件循环暂停当前协程执行，但**不会**切换到其他任务。只有当 `await` 后面是 `future` 对象时，事件循环才会真正切换任务。

```python
import asyncio

async def task02():
    print('task02')
    return 200

async def task01():
    print('task01')
    await asyncio.sleep(0)  # sleep 内部会 await future，触发任务切换
    print('task01 继续')
    return 100

async def start():
    result = await asyncio.gather(task01(), task02())
    print(result)

asyncio.run(start())
```

## 四、Future 对象深入

Future 对象是异步编程的核心概念，它是异步执行等待任务的对象和事件循环之间的**桥梁**。

![Future 对象工作原理](/images/python-async-future.png)

### 4.1 Future 的作用

假设事件循环管理两个任务，执行过程如下：

1. 当事件循环执行到 **task1 -> sub_task -> 网络接收数据** 时，需要等待
2. 创建一个新线程，把该任务扔到该线程中执行
3. 创建一个 future 对象，让事件循环和新线程共同持有
4. 当新线程接收到数据后，通过 future 对象将数据传递到 task1
5. 事件循环检测到 future 有返回结果，继续调度 task1 执行
6. 由于 task1 已被挂起，事件循环切换到 task2 执行

### 4.2 Future 实例

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

def thread_task(future):
    time.sleep(5)
    future.set_result(100)

async def sub_task():
    print('sub task 开始')

    event_loop = asyncio.get_running_loop()
    future = event_loop.create_future()
    executor = ThreadPoolExecutor()

    # 在其他线程执行任务
    event_loop.run_in_executor(executor, thread_task, future)
    # 挂起当前任务，等待 future 结果
    result = await future

    print('sub task 结束')
    return result

async def task1():
    print('task1 开始')
    result = await sub_task()
    print('task1 结束')
    return result

async def task2():
    print('task2 开始')
    await asyncio.sleep(1)
    print('task2 结束')
    return 200

async def main():
    result = await asyncio.gather(task1(), task2())
    print(result)

asyncio.run(main())
```

**执行结果：**
```
task1 开始
sub task 开始
task2 开始
task2 结束
sub task 结束
task1 结束
[100, 200]
```

## 五、总结

异步编程是一种在**单线程中实现并发执行**的技术，它通过"事件循环"来调度任务，在遇到等待操作时将控制权交给其他任务，避免空等，大大提升运行效率。

### 核心要点

| 概念 | 说明 |
|------|------|
| **async def** | 定义协程函数，返回协程对象 |
| **await** | 挂起当前任务，等待异步操作完成 |
| **事件循环** | 负责任务调度和执行的核心机制 |
| **Future** | 连接异步操作和事件循环的桥梁 |

### 适用场景

异步编程在以下场景最为有效：

- 文件读写操作频繁
- 网络请求密集
- 数据库操作耗时较长

### 编程思维转变

写代码时需要具备"异步思维"：

- 凡是会等待的操作，都应考虑是否使用 `await`
- 每写一个任务，自问：**这个地方是不是可以不阻塞？**
- 关注**是否等待**，主动识别可异步处理的环节

> **一句话总结**：异步编程就是让单线程也能干多件事，不等白等，高效非阻塞。

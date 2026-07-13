---
title: "从零训练大模型（三）：PyTorch 极简入门"
author: Aidenz
pubDatetime: 2026-07-13T12:15:00Z
slug: train-llm-scratch-03-pytorch-intro
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 3
tags:
  - LLM
  - 大模型
  - 从零训练
  - PyTorch
description: "PyTorch 极简入门：张量运算、自动求导、nn.Module，最后亲手写一个最小可跑的训练循环，把前两章的概念落到代码。"
---

> **本章前置**:读完[第 01 章](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-01-setup)(环境已装好,`torch` 可用)和[第 02 章](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-02-math-ml-basics)(理解张量形状、梯度、链式法则、以及一维线性回归那套"训练核心循环")。
>
> **你将学到**:确认 PyTorch 装好;创建张量、看懂它的形状/`dtype`/`device`、用一用"广播";理解**自动求导(autograd)**——`requires_grad`、`.backward()`、`.grad`,并把第 02 章手推的线性回归梯度,和 autograd 自动算出的梯度**对照验证**(这会是你的"啊哈"时刻);认识 `nn.Module`、`nn.Linear`、损失函数、`torch.optim`;最后**亲手写一个最小但完整的训练循环**拟合 $y = 2x$,看着 loss 一步步下降。我们会把"训练循环五件套"明确点出来——记住它,后面所有阶段的骨架都是这五步。
>
> 👈 [上一章:数学与机器学习最小基础](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-02-math-ml-basics) ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

上一章我们在纸上手推了梯度、手动更新了参数,挺累的吧?这一章你会发现:那些苦活,PyTorch 全都自动帮你干了。PyTorch 是这个项目(以及当今绝大多数大模型)所用的深度学习框架。它给你两样最重要的东西:**张量**(能在 CPU/GPU 上高效计算的多维数组)和 **autograd**(自动求导——你只管写前向计算,它自动把所有梯度算出来)。

本章全部代码都能在 **CPU 上运行**,跟着敲一遍,理解会牢得多。建议你打开一个 Python 交互窗口(终端里输入 `python3`),边读边试。

## 1. 确认 PyTorch 装好了

第 01 章装项目时已经把 `torch` 装上了。先确认一下。在已激活虚拟环境的终端里:

```python
import torch
print(torch.__version__)        # 打印 PyTorch 版本号
print(torch.cuda.is_available())  # 有没有可用的 NVIDIA GPU(没有就是 False,不影响本章)
```

能打印出版本号(比如 `2.x.x`)就说明一切就绪。`torch.cuda.is_available()` 返回 `False` 完全没关系——本章不需要 GPU。

## 2. 张量:PyTorch 里的"万物"

第 02 章说过,机器学习里的数据都是张量。在 PyTorch 里,张量就是 `torch.Tensor`。我们看看怎么造、怎么看它的属性。

### 创建张量

```python
import torch

# 从 Python 列表直接造
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(a)
print(a.shape)   # torch.Size([2, 3]) —— 2 行 3 列

# 全 0、全 1
z = torch.zeros(2, 3)   # 形状 (2, 3) 的全 0 张量
o = torch.ones(3)       # 长度 3 的全 1 向量

# 随机张量(标准正态分布)
r = torch.randn(2, 16, 128)   # 形状 (2, 16, 128) —— 还记得 (B, T, C) 吗?

# 随机整数(常用来造假的 token id),范围 [0, 50304)
idx = torch.randint(0, 50304, (2, 16))   # 形状 (2, 16)
```

### 形状(shape)

`shape` 是张量最重要的属性,告诉你每一维多大——第 02 章已经反复强调过。随手 `print(x.shape)` 是写深度学习代码时最常用的调试手段,养成习惯:**形状对不上,十有八九就是 bug**。

### dtype:数据类型

`dtype` 指张量里数字的类型。最常见两类:

```python
f = torch.tensor([1.0, 2.0])     # 默认 torch.float32(浮点数,带小数)
i = torch.tensor([1, 2, 3])      # 默认 torch.int64(整数)
print(f.dtype, i.dtype)
```

- **浮点数(`float32` 等)**:用于权重、激活值、loss——一切需要"带小数、能求梯度"的量。
- **整数(`int64`/`long`)**:用于 token id、索引这种"必须是整数"的量。第 01 章喂给模型的 `idx` 就是整数张量。

> 你可能在项目里见过 `bf16`(bfloat16)这种类型——那是为了在 GPU 上省显存、加速的"混合精度",第 08 章会讲。本章只用默认的 `float32`。

### device:张量住在哪

`device` 表示张量在哪儿计算:`cpu`(处理器)还是 `cuda`(NVIDIA 显卡)。

```python
x = torch.randn(3)
print(x.device)          # cpu(默认)
# 如果有 GPU,可以搬过去:x = x.to("cuda")
```

本章一律用 CPU。记住一条规则:**参与同一次运算的张量,必须在同一个 device 上**,否则报错。第 01 章 smoke 配置里那个 `device: "cpu"` 就是在统一指定这件事。

### 广播(broadcasting):形状不同也能逐元素运算

经常会遇到"形状不完全一样的两个张量要做加减乘"。**广播**是 PyTorch 自动把它们"对齐"的规则:把较小的那个沿着缺的维度"复制扩展",好让逐元素运算成立——但**不会真的占用额外内存**,只是逻辑上扩展。

最常见的例子:一个张量 + 一个标量。

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(x + 10)        # tensor([11., 12., 13.]) —— 标量 10 被广播到每个元素
print(x * 2)         # tensor([2., 4., 6.])
```

稍复杂一点:矩阵的每一行加同一个向量。

```python
m = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])      # 形状 (2, 3)
bias = torch.tensor([10.0, 20.0, 30.0])  # 形状 (3,)
print(m + bias)
# tensor([[11., 22., 33.],
#         [14., 25., 36.]])
# bias 被广播到 m 的每一行
```

广播规则的直觉:**从最后一维开始往前对齐,某一维要么相等、要么其中一个是 1(或不存在),就能广播。** 神经网络里给每个特征加偏置(bias)、做归一化,背后都是广播。理解它,以后看代码里"形状不一样却能直接相加"就不会迷惑了。

## 3. autograd:让 PyTorch 替你算梯度

这是 PyTorch 最神奇、也最省事的部分。第 02 章你手推了 $\frac{\partial L}{\partial w}$、$\frac{\partial L}{\partial b}$,还得小心链式法则。**autograd 会自动、精确地做完这一切**——你只写前向计算,它在背后记录每一步,然后一声令下就把所有梯度反向传播出来。

### 三个关键:`requires_grad`、`.backward()`、`.grad`

- **`requires_grad=True`**:告诉 PyTorch "请追踪对这个张量的所有运算,我之后要它的梯度"。模型参数都开着这个开关。
- **`.backward()`**:对最终的标量(通常是 loss)调用它,触发反向传播,自动算出所有梯度。
- **`.grad`**:反向传播后,每个 `requires_grad=True` 的张量身上会多出一个 `.grad` 属性,存着"loss 对它的梯度"。

看个最小的例子,验证一下第 02 章学过的 $f(x) = x^2$,导数 $f'(x) = 2x$,在 $x=3$ 处应为 6:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)  # 追踪对 x 的运算
y = x ** 2                                  # 前向:y = x^2
y.backward()                                # 反向:自动求 dy/dx
print(x.grad)                               # tensor(6.) —— 正是 2*3 = 6!
```

是不是很爽?你没写任何求导公式,PyTorch 自己算出了 `6`。

### "啊哈"时刻:autograd 和你的手推梯度完全一致

现在把第 02 章那个线性回归的例子搬过来,**用 autograd 重算一遍梯度,和你手推的结果对照**。

回忆第 02 章的设定:模型 $\hat y = wx + b$,单点 $(x, y) = (1, 2)$,初始 $w = 0, b = 0$。我们手推过这一步的梯度是 $\frac{\partial L}{\partial w} = -4$、$\frac{\partial L}{\partial b} = -4$。验证:

```python
import torch

# 参数,开启梯度追踪
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 数据点
x = torch.tensor(1.0)
y = torch.tensor(2.0)

# 前向:预测 + 均方误差损失(单点)
y_hat = w * x + b           # = 0
loss = (y_hat - y) ** 2     # = (0 - 2)^2 = 4

# 反向:autograd 自动算梯度
loss.backward()

print("dL/dw =", w.grad.item())   # -4.0
print("dL/db =", b.grad.item())   # -4.0
```

输出:

```
dL/dw = -4.0
dL/db = -4.0
```

**和你第 02 章手推的 $-4$、$-4$ 一模一样!** 这就是 autograd 的承诺:无论模型多复杂、有多少层多少参数,你只要写出"前向怎么算 loss",`.backward()` 就替你把所有梯度精确求出来。第 02 章那套累人的链式法则手推,以后交给它就行——你只需理解原理、相信结果。

> **小细节**:`.item()` 把"只含一个数的张量"变成一个普通 Python 数,方便打印。另外,梯度会在每次 `.backward()` 时**累加**到 `.grad` 上,而不是覆盖——所以训练循环里每步更新完都要"清零梯度",这一点第 5 节会专门讲。

## 4. 搭积木的工具:nn.Module、nn.Linear、损失函数、优化器

手动管理一堆 `w`、`b` 很快就会失控(想象一个有上亿参数的模型)。PyTorch 提供了一套"搭积木"的工具,帮你优雅地组织参数和计算。

### `nn.Module`:一切模型的基类

PyTorch 里所有模型都继承自 `nn.Module`。它帮你**自动管理参数**(自动开 `requires_grad`、自动收集所有参数供优化器更新)。第 01 章见过的 `Transformer` 类,以及 `src/models/transformer.py` 里的所有组件,都是 `nn.Module` 的子类——你已经用过它了,只是当时没细说。

### `nn.Linear`:最常用的一块积木

`nn.Linear(in_features, out_features)` 就是第 02 章说的"矩阵乘法 + 偏置":把一个 `in_features` 维的输入,线性变换成 `out_features` 维的输出。内部它自己持有权重矩阵 $W$ 和偏置向量 $b$,做的就是 $y = xW^\top + b$。

```python
import torch
import torch.nn as nn

linear = nn.Linear(in_features=128, out_features=512)
x = torch.randn(2, 16, 128)   # (B, T, C=128)
out = linear(x)
print(out.shape)              # torch.Size([2, 16, 512]) —— 最后一维从 128 变成 512
```

整个 Transformer,说到底就是一堆 `nn.Linear`(加上注意力、归一化等)按精巧的方式堆起来的。第 05 章你会看到 `lm_head` 就是一个 `nn.Linear(n_embed, vocab_size)`。

### 损失函数

PyTorch 内置了常用损失。第 02 章我们手写过均方误差;PyTorch 里直接用 `nn.MSELoss()`。语言模型用的交叉熵则是 `nn.CrossEntropyLoss()`(或函数式的 `F.cross_entropy`)——第 01 章读到的 `transformer.py` 里那行 `F.cross_entropy(...)` 就是它,第 07 章会完整推导。

```python
import torch.nn as nn
mse = nn.MSELoss()
pred = torch.tensor([0.0, 0.8])
target = torch.tensor([2.0, 2.0])
print(mse(pred, target))   # 自动算 ((0-2)^2 + (0.8-2)^2) / 2
```

### 优化器 `torch.optim`

第 02 章里,我们手动写 `w ← w - η·grad` 来更新参数。**优化器**把这一步自动化、还做得更聪明。最基础的是 `torch.optim.SGD`(随机梯度下降,就是第 02 章那条朴素更新规则);更常用、更强的是 `Adam` / `AdamW`,它会自适应地调整每个参数的步长,通常收敛更快更稳。

```python
import torch
# 假设 params 是模型的参数
optimizer = torch.optim.SGD(params, lr=0.1)     # 朴素梯度下降,学习率 0.1
# 或者:
optimizer = torch.optim.Adam(params, lr=1e-3)   # 更聪明的优化器
```

> **呼应仓库**:本项目的后训练阶段用的就是 **AdamW**。你可以在 `src/post_training/optim.py` 里看到 `torch.optim.AdamW(...)` 的真实调用(它还做了"只给二维以上的权重加权重衰减"这种工程细节)。AdamW 的来龙去脉(从梯度下降 → Adam → AdamW)第 08 章会讲透。本章你只需知道:**优化器 = 帮你自动、聪明地执行"沿梯度反方向更新参数"这一步**。

## 5. 亲手写一个最小完整训练循环:拟合 y = 2x

工具齐了,我们把它们组装成第 02 章描述的那套"训练核心循环",让一个模型**自己学出** $y = 2x$ 这条规律。这一节是本章的高潮——跑通它,你就真正理解"训练"在代码里长什么样。

完整脚本如下(可直接保存为 `train_line.py` 运行):

```python
import torch
import torch.nn as nn

# --- 1. 造点数据:真实规律是 y = 2x(模型并不知道,要自己学) ---
x = torch.randn(64, 1)        # 64 个样本,每个 1 维特征,形状 (64, 1)
y = 2 * x                     # 对应的真实标签

# --- 2. 定义模型:一个 1 维输入、1 维输出的线性层(就是 y_hat = w*x + b) ---
model = nn.Linear(1, 1)

# --- 3. 损失函数 + 优化器 ---
loss_fn = nn.MSELoss()                              # 均方误差(第 02 章手算过)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 朴素梯度下降

# --- 4. 训练循环:把"五件套"重复很多次 ---
for step in range(200):
    # (1) 前向:用当前参数做预测
    y_hat = model(x)
    # (2) 算 loss:预测和真值差多少
    loss = loss_fn(y_hat, y)
    # (3) 反向:autograd 自动算出所有梯度,存进每个参数的 .grad
    loss.backward()
    # (4) 更新:优化器沿梯度反方向把参数挪一小步
    optimizer.step()
    # (5) 清零梯度:为下一步做准备(否则梯度会累加,见第 3 节那条小细节)
    optimizer.zero_grad()

    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

# --- 训练完,看看模型学到的参数,应该接近 w=2, b=0 ---
w = model.weight.item()
b = model.bias.item()
print(f"\n学到的 w = {w:.3f}(目标 2.0),b = {b:.3f}(目标 0.0)")
```

运行 `python3 train_line.py`,你会看到 loss **一步步下降**,类似:

```
step   0  loss 3.9xxx
step  20  loss 0.6xxx
step  40  loss 0.1xxx
...
step 180  loss 0.0000

学到的 w = 2.000(目标 2.0),b = 0.000(目标 0.0)
```

(具体数字因随机初始化略有不同,但趋势一定是 loss 降到接近 0、`w` 逼近 2、`b` 逼近 0。)

**这就是"训练"在代码里的全部样子。** 模型从一个随机的 `w`、`b` 出发,靠"前向→算 loss→反向→更新→清零"这套循环,反复地把参数往"损失更小"的方向挪,最终自己发现了 $y = 2x$ 这条规律。和第 02 章那个手算例子是同一回事,只是这次梯度由 autograd 算、更新由优化器做,而且一口气跑了 200 步。

### 训练循环五件套(请刻在脑子里)

把上面循环体里的五步抽出来,这就是**贯穿全课的骨架**:

1. **前向(forward)**:`y_hat = model(x)` —— 用当前参数算出预测。
2. **算损失(loss)**:`loss = loss_fn(y_hat, y)` —— 衡量预测有多差。
3. **反向(backward)**:`loss.backward()` —— autograd 自动求出所有梯度。
4. **更新(step)**:`optimizer.step()` —— 沿梯度反方向挪一步。
5. **清零(zero_grad)**:`optimizer.zero_grad()` —— 清掉这步的梯度,准备下一轮。

**记住这五步。** 后面无论是预训练(第 11 章)、SFT 指令微调(第 12 章),还是更复杂的 PPO、GRPO(第 15、16 章),核心训练循环**都是这同一个骨架**。变的只是:模型更大(从 `nn.Linear(1,1)` 变成上亿参数的 Transformer)、数据更复杂(从 $y=2x$ 变成海量文本或人类偏好)、损失更精巧(交叉熵、偏好损失、带裁剪的策略目标……)、优化器更强(AdamW)。但"前向→loss→backward→step→zero_grad"这五个字,从头到尾不变。你今天写的这 20 行,就是一切大模型训练的"细胞核"。

> **小提醒:`zero_grad` 的位置**。把它放在 `step()` 之后(下一轮 `backward()` 之前)即可;有些代码写在循环开头,效果一样。关键是**每轮都要清一次**,否则梯度会跨轮累加,导致更新错误。第 3 节解释过"梯度默认累加"的原因——这个"特性"在第 08 章讲"梯度累积"时反而会被巧妙利用,但在这里我们只需老老实实清零。

## 小结

- **张量**是 PyTorch 的核心数据结构;牢记看它的 **`shape`**(形状)、**`dtype`**(浮点 vs 整数)、**`device`**(CPU/GPU);**广播**让形状不同的张量也能逐元素运算(如给每行加偏置)。
- **autograd** 自动求导:给张量开 `requires_grad=True`,对 loss 调 `.backward()`,梯度就出现在各张量的 `.grad` 里。我们验证了它和第 02 章手推的线性回归梯度**完全一致**。
- 搭积木工具:**`nn.Module`**(模型基类、自动管理参数)、**`nn.Linear`**(矩阵乘法+偏置)、损失函数(`nn.MSELoss`、`nn.CrossEntropyLoss`)、**`torch.optim`**(SGD / Adam / AdamW,自动执行参数更新)。本项目后训练用的是 `AdamW`(见 `src/post_training/optim.py`)。
- 我们亲手写了**最小完整训练循环**拟合 $y=2x$,看着 loss 降到接近 0、参数逼近真值。
- **训练循环五件套**:前向 → 算 loss → backward → step → zero_grad。这是全课所有训练阶段共用的骨架,务必记牢。

## 自测题

1. **`requires_grad=True`、`.backward()`、`.grad` 三者在自动求导里各扮演什么角色?**
   <details><summary>提示 / 答案</summary>`requires_grad=True` 让 PyTorch 追踪对该张量的运算;对标量 loss 调 `.backward()` 触发反向传播、自动算梯度;算完后梯度存在该张量的 `.grad` 属性里。</details>

2. **训练循环里如果忘了写 `optimizer.zero_grad()`,会发生什么?为什么?**
   <details><summary>提示 / 答案</summary>梯度默认是**累加**而非覆盖的,不清零会导致每一步的梯度叠加上之前所有步的梯度,更新方向和幅度都错,训练发散或乱跑。所以每轮都要清零。</details>

3. **`nn.Linear(128, 512)` 接收一个形状 `(2, 16, 128)` 的输入,输出形状是多少?它内部本质在做什么运算(回忆第 02 章)?**
   <details><summary>提示 / 答案</summary>输出 `(2, 16, 512)`(只改最后一维 128→512)。本质是"矩阵乘法 + 偏置":$y = xW^\top + b$,把每个 token 的 128 维特征线性变换成 512 维。</details>

4. **完整写出"训练循环五件套"的五个步骤(代码或文字皆可),并说明哪一步用到了 autograd。**
   <details><summary>提示 / 答案</summary>(1) 前向 `y_hat = model(x)`;(2) 算损失 `loss = loss_fn(y_hat, y)`;(3) 反向 `loss.backward()`(这一步用 autograd 自动求梯度);(4) 更新 `optimizer.step()`;(5) 清零 `optimizer.zero_grad()`。</details>

5. **本项目后训练阶段用的优化器是什么?在哪个文件里能找到它的真实调用?**
   <details><summary>提示 / 答案</summary>是 **AdamW**(`torch.optim.AdamW`),真实调用在 `src/post_training/optim.py` 里(它还按"二维以上权重才加权重衰减"的规则分组)。从梯度下降到 AdamW 的演进会在第 08 章讲。</details>

## 深入参考

- 优化器从 SGD → Adam → AdamW 的演进、学习率调度、梯度累积、混合精度,见本教程 [第 08 章](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-08-optimization)。
- 交叉熵损失的完整推导,见 [第 07 章](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-07-objectives)。
- 真实的优化器构造代码:`src/post_training/optim.py`;真实的模型与前向/损失代码:`src/models/transformer.py`。
- 工程速查参考:`../foundations/` 下的基础文档(随课程推进对照)。

---

地基三章到此打完——你已经会搭环境、懂了必要的数学、能用 PyTorch 写出训练循环。从下一章开始,我们正式进入语言模型的核心:先看**文本是怎么变成数字**的,以及数据被整理成什么形状喂给模型。

下一章 👉 [第 04 章:文本如何变成数字:分词与数据形状](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-04-tokenization)

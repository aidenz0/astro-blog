---
title: "从零训练大模型（七）：训练目标——交叉熵与困惑度"
author: Aidenz
pubDatetime: 2026-07-13T08:11:00Z
slug: train-llm-scratch-07-objectives
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 7
tags:
  - LLM
  - 大模型
  - 从零训练
  - 损失函数
description: "训练目标的完整推导：从最大似然到负对数似然再到交叉熵，讲清困惑度（perplexity）的含义，以及标签错位与掩码在实现中的细节。"
---

> **本章前置**:你已读过第 01–06 章。也就是说你知道:对数与概率的基本运算、softmax 把 logits 变成概率分布、Transformer 在每个位置输出一个长度为 $V$(词表大小)的 logits 向量、以及数据被打包成形状 $(B,T)$ 的 token 批次。
>
> **你将学到**:模型到底**根据什么信号学习**。我们从"语言模型就是一串条件概率的连乘"出发,经由**最大似然估计(MLE)→ 负对数似然(NLL)→ 交叉熵(cross-entropy)**一步步推导出训练损失;手推交叉熵对 logits 的梯度 $p-\text{onehot}(y)$,搞懂"为什么交叉熵这么好训";弄清输入和标签**错位一位(shift)**的具体下标;推导**困惑度(perplexity)**并理解它的直觉;最后看 SFT 里如何用**掩码损失**只对 assistant 的回答算损失(为第 12 章埋伏笔)。动手部分用一个 3 类小例子手算交叉熵,并用 `torch.nn.functional.cross_entropy` 对照。
>
> 👈 [上一章:注意力机制 · 完整推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-06-attention) ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

## 7.1 语言模型就是一串条件概率

我们想让模型学会"语言"。"会语言"用数学说,就是模型能对**任意一句话出现的概率**给出一个合理的估计——常见的话概率高,胡言乱语概率低。

设一句话(一个序列)由 $T$ 个 token 组成:$x_1, x_2, \ldots, x_T$。整句话的概率,可以用概率的**链式法则**一字不漏地拆成连乘:

$$
p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
$$

逐符号解释:$p_\theta$ 表示"由参数为 $\theta$(模型所有权重)的模型给出的概率";$x_{1:T}$ 是整句话;$\prod$ 是连乘;$x_{<t}$ 表示"$x_t$ 之前的所有 token"(即 $x_1,\ldots,x_{t-1}$)。这个式子的大白话是:**一句话的概率 = 第 1 个 token 出现的概率 × (在第 1 个已知的前提下第 2 个的概率) × (在前 2 个已知的前提下第 3 个的概率) × ……**。

为什么能这样拆?这就是链式法则,永远成立($P(AB)=P(A)P(B\mid A)$ 的多变量推广),不需要任何假设。它之所以重要,是因为它把"建模一整句话"这个大问题,**化简成了一个反复出现的小问题:给定前文,预测下一个 token**。而"给定前文预测下一个 token"——正是我们第 06 章那个带因果掩码的 Transformer 干的事!每个位置 $t$ 输出的 logits,经过 softmax,就是 $p_\theta(x_{t+1}\mid x_{\le t})$ 这个分布。一切都对上了。

---

## 7.2 从最大似然到负对数似然

### 7.2.1 最大似然估计(MLE):让真实数据"最不意外"

我们手上有一大堆真实文本(训练语料)。训练的目标朴素地讲就是:**调整参数 $\theta$,让模型觉得"这些真实文本"出现的概率尽可能大**。模型如果给真实文本打了高概率,说明它"觉得真实的句子很正常、不意外",这正是我们想要的。这个原则叫**最大似然估计**(Maximum Likelihood Estimation,MLE)。

对单个序列,我们要最大化的就是 7.1 的 $p_\theta(x_{1:T})$:

$$
\theta^\star = \arg\max_\theta \; p_\theta(x_{1:T}) = \arg\max_\theta \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
$$

逐符号解释:$\arg\max_\theta$ 的意思是"找到让后面那个式子取最大值的 $\theta$"。也就是说,我们要找一组权重,使模型赋予真实序列的概率最大。

### 7.2.2 取对数:把连乘变成连加

直接对一长串概率的**连乘**做优化非常糟糕:每个概率都是 0 到 1 之间的小数,几百上千个一连乘,结果会小到计算机的浮点数都表示不了(数值下溢)。救星是**对数**。对数有个黄金性质:$\log(ab)=\log a+\log b$,它能把连乘变成连加。而且对数是**单调递增**函数——最大化 $p$ 和最大化 $\log p$ 得到的 $\theta^\star$ 完全一样,不改变答案。于是两边取对数:

$$
\log p_\theta(x_{1:T}) = \log \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}) = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

逐句解释:连乘的对数,等于各项对数的连加。右边这个 $\sum_t \log p_\theta(\cdots)$ 叫**对数似然**(log-likelihood)。它在数值上稳定得多,也方便求导(加法的导数好算)。

### 7.2.3 取负:把"最大化"变成"最小化"损失

机器学习的惯例是**最小化一个损失函数**(loss,越小越好),而不是最大化。这只需在对数似然前面加个负号,得到**负对数似然**(Negative Log-Likelihood,NLL):

$$
\mathcal{L}_{\text{NLL}} = -\log p_\theta(x_{1:T}) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

逐句解释:加负号后,"最大化对数似然"就等价于"**最小化负对数似然**"。这就是我们的损失函数雏形。直觉上:如果模型给真实 token 的概率 $p$ 很接近 1,$\log p$ 接近 0,$-\log p$ 也接近 0,损失小(好);如果模型给真实 token 的概率很小(接近 0),$-\log p$ 会冲向 $+\infty$,损失巨大(差)。**模型越是"对真实答案感到意外",罚得越狠**,这非常合理。

到这里请记住这条主线:**最大化似然 = 最小化 NLL**。接下来我们说明,单个位置的 NLL 其实就是大名鼎鼎的"交叉熵"。

---

## 7.3 单个位置:交叉熵就是 $-\log$ 真实 token 的概率

聚焦某一个位置 $t$。模型在这里输出一个长度为 $V$ 的 logits 向量 $z \in \mathbb{R}^V$($V$ 是词表大小,本仓库约 5 万)。softmax 把它变成在整个词表上的概率分布:

$$
p_i = \operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}, \qquad i = 1,\ldots,V
$$

逐符号解释:$z_i$ 是 logits 的第 $i$ 个分量(对应词表里第 $i$ 个 token 的"原始得分");$p_i$ 是模型认为"下一个 token 是第 $i$ 个词"的概率。所有 $p_i$ 加起来等于 1。

真实的下一个 token 是某个确定的 id,记为 $y$。我们可以把这个"标准答案"写成一个 **one-hot 向量** $\mathbf{1}_y$:它只在第 $y$ 个位置是 1,其余全是 0(意思是"正确答案 100% 是第 $y$ 个词")。**交叉熵**衡量"真实分布 $\mathbf{1}_y$"和"模型分布 $p$"之间的差距,定义为:

$$
\text{CE}(p, y) = -\sum_{i=1}^{V} (\mathbf{1}_y)_i \, \log p_i
$$

逐符号解释:对词表里每个词,把"真实分布在该词上的值"乘以"模型概率的对数",求和再取负。但 $\mathbf{1}_y$ 只有第 $y$ 个位置是 1、别的都是 0,所以这个和里**只有 $i=y$ 这一项活下来**,其余全乘以 0 消失了:

$$
\boxed{\;\text{CE}(p, y) = -\log p_y = -\log \operatorname{softmax}(z)_y\;}
$$

逐句解释:交叉熵化简成了非常干净的一句话——**就是"真实那个 token 的概率取对数,再加负号"**。这恰好就是 7.2 里单个位置的负对数似然!所以在语言模型里,"交叉熵损失"和"负对数似然"是同一个东西的两个名字。模型给正确 token 的概率越高,这一项损失越小。

### 7.3.1 关键推导:交叉熵对 logits 的梯度是 $p - \text{onehot}(y)$

为什么交叉熵配 softmax"特别好训"?秘密在它的**梯度**异常干净。我们要算损失 $\text{CE}=-\log p_y$ 对每个 logit $z_k$ 的偏导。这一步是本章的数学高潮,我们一步不跳地推。

先把交叉熵用 logits 完整写开。因为 $p_y = \dfrac{\exp(z_y)}{\sum_j \exp(z_j)}$,取对数:

$$
\text{CE} = -\log p_y = -z_y + \log \sum_{j=1}^{V}\exp(z_j)
$$

逐句解释:$\log$ 把分式变成"分子的对数减分母的对数",即 $\log\exp(z_y)-\log\sum_j\exp(z_j) = z_y - \log\sum_j\exp(z_j)$,再整体取负就得到上式。记右边那个 $\log\sum_j \exp(z_j)$ 为 $\text{LSE}$(log-sum-exp)。

现在对某个 $z_k$ 求偏导,分两部分。

**第一部分**,$-z_y$ 对 $z_k$ 求导:只有当 $k=y$ 时它是 $-1$,否则是 $0$。用 one-hot 记号写,就是 $-(\mathbf{1}_y)_k$。

**第二部分**,$\text{LSE}=\log\sum_j\exp(z_j)$ 对 $z_k$ 求导。用链式法则(外层 $\log u$ 的导数是 $1/u$,内层 $\sum_j\exp(z_j)$ 对 $z_k$ 的导数只有 $j=k$ 那一项 $\exp(z_k)$ 留下):

$$
\frac{\partial}{\partial z_k}\log\sum_{j}\exp(z_j) = \frac{\exp(z_k)}{\sum_{j}\exp(z_j)} = p_k
$$

逐句解释:这正好又是 softmax 的定义!$\text{LSE}$ 对第 $k$ 个 logit 的导数,恰好等于模型给第 $k$ 个词的概率 $p_k$。

把两部分合起来:

$$
\boxed{\;\frac{\partial\,\text{CE}}{\partial z_k} = p_k - (\mathbf{1}_y)_k\;}
\qquad\Longleftrightarrow\qquad
\frac{\partial\,\text{CE}}{\partial z} = p - \text{onehot}(y)
$$

逐句解释这为什么是"好训"的关键:梯度就是**"模型预测的概率分布"减去"真实的 one-hot 分布"**,干净得不能再干净。它的含义极其直观——

- 对**真实**那个 token($k=y$):梯度是 $p_y - 1$,是个负数。梯度下降会沿梯度反方向走,于是它**抬高** $z_y$,让模型更倾向选真实 token。模型当前越没把握($p_y$ 离 1 越远),这股推力越大。
- 对**其他**错误 token($k\ne y$):梯度是 $p_k - 0 = p_k$,是个正数,于是它**压低** $z_k$。模型当前给某个错词的概率越高,压它的力越大。
- 当模型完全预测对了($p=\text{onehot}(y)$),梯度处处为 0,不再调整。

没有讨厌的饱和、没有梯度消失、推力大小自动正比于"错得多严重"。这就是交叉熵 + softmax 成为分类/语言建模标配的根本原因。

---

## 7.4 整段序列与一个 batch 的平均损失

7.3 处理的是一个位置。一条序列有 $T$ 个位置、一个 batch 有 $B$ 条序列。我们把所有位置的交叉熵**取平均**作为最终损失(取平均而非求和,是为了让损失值不随 batch 大小、序列长度变化,便于比较):

$$
\mathcal{L}_{\text{LM}} = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\Big(-\log p_\theta\big(y_{b,t}\mid x_{b,\le t}\big)\Big)
$$

逐符号解释:外层两个求和遍历 batch 里每条序列 $b$ 和每个位置 $t$;括号里是该位置的交叉熵($y_{b,t}$ 是这条序列在位置 $t$ 的真实下一个 token);前面的 $\frac{1}{BT}$ 求平均。一句话:**整批数据上,每个位置交叉熵的平均值**。

### 7.4.1 输入与标签错位一位(shift)的具体下标

上面式子里反复出现"位置 $t$ 的真实下一个 token $y_t$"。这"下一个"在工程上就是一次**错位一位**(shift by one)。具体地,给模型的输入和它要预测的标签是同一串 token,只是错开一格:

$$
\text{输入 } x = [t_0, t_1, t_2, \ldots, t_{T-1}]
$$

$$
\text{标签 } y = [t_1, t_2, t_3, \ldots, t_{T}]
$$

逐句解释下标对应:模型在看到 $t_0$(输入第 0 位)时,要预测的标签是 $t_1$(标签第 0 位);看到前缀 $[t_0,t_1]$ 时要预测 $t_2$;……一句话:**标签就是输入整体往左挪一位**。位置 $t$ 的输入是 $x_t=t_t$,它要预测的目标是 $y_t=t_{t+1}$,这正是"用前文预测下一个 token"。

在本仓库里,这个错位有两种写法,理解上等价。预训练主路径 `src/models/transformer.py` 的 `forward` 里,数据加载时**已经**把 `targets` 准备成"比 `idx` 超前一位"的张量了,所以 `forward` 直接在所有位置上一把算交叉熵,不必在函数里再手动错位:

```python
x = self.forward_hidden(idx)
logits = self.lm_head(x)
loss = None
if targets is not None:
    B, T, C = logits.shape
    flat_logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T).long()
    loss = F.cross_entropy(flat_logits, targets)
return logits, loss
```

逐行解释:`logits` 形状是 $(B,T,V)$(这里变量名 `C` 实为词表大小 $V$);`reshape(B * T, C)` 把前两维 $B,T$ 拍平成一长条,变成 $(BT, V)$;`targets.reshape(B * T)` 把标签也拍平成 $(BT,)$ 的整数向量;然后 `F.cross_entropy` 一次性对这 $BT$ 个位置算交叉熵并求平均——这正是 $\mathcal{L}_{\text{LM}}$。(顺带一提:注释里特意用 `reshape` 而不是 `view`,是因为 `targets` 来自一个非连续的张量切片,在 CPU 上 `view` 会报错,`reshape` 更稳妥。)

而 SFT 路径 `src/post_training/sft.py` 则把错位**显式地写出来**(因为它还要配一个掩码,见 7.6 节):

```python
# Predict token t+1 from position t (same shift the base model uses).
logits = logits[:, :-1, :]
targets = tokens[:, 1:]
```

逐行解释:`logits[:, :-1, :]` 取**除最后一个位置外**的所有位置(因为最后一个位置没有"下一个 token"可预测,丢掉);`tokens[:, 1:]` 取**从第二个 token 开始**的序列作为标签。这两行合起来,就把"位置 $t$ 的输出"和"第 $t+1$ 个真实 token"严丝合缝地对齐了——和上面那张错位下标表完全一致。

---

## 7.5 困惑度:平均每步在多少个选项里纠结

损失值 $\mathcal{L}$(平均交叉熵)是个抽象的数,不太好"体感"。**困惑度**(perplexity,PPL)给它换了个更直观的尺度。定义就是把平均交叉熵放到指数上:

$$
\boxed{\;\text{PPL} = \exp(\mathcal{L}) = \exp\!\Big(-\tfrac{1}{BT}\textstyle\sum_{b,t}\log p_\theta(y_{b,t}\mid x_{b,\le t})\Big)\;}
$$

逐符号解释:$\exp$ 是以 $e$ 为底的指数函数,正好"抵消"交叉熵里的自然对数 $\log$。

为什么这样定义有意义?我们从直觉推一遍。先看单个位置:若模型给真实 token 的概率是 $p$,那一步的交叉熵是 $-\log p$,它的困惑度是 $\exp(-\log p)=\frac{1}{p}$。也就是说,**单步困惑度 = 真实 token 概率的倒数**。如果模型很确定($p=1$),困惑度是 1(一点都不困惑);如果模型在 $k$ 个候选里完全均匀地猜(每个概率 $1/k$),困惑度就是 $k$。所以困惑度的直觉是:

> **平均每预测一个 token,模型大约在多少个"等可能的选项"之间纠结。**

困惑度越低越好:1 是完美(每步都笃定),越大说明模型越"懵"。再看一个有用的标尺——一个**完全没训练、瞎猜**的模型,会给词表里每个 token 大致均等的概率 $1/V$,此时平均交叉熵约为 $\log V$,困惑度约为

$$
\exp(\log V) = V
$$

也就是"在整个词表 $V$ 个词里均匀乱猜"。本仓库词表 $V \approx 50304$,对应初始交叉熵约 $\log(50304)\approx 10.83$。所以你训练时如果看到 loss 从约 10.8 开始往下掉、困惑度从约 5 万往下掉,就说明模型正在从"瞎猜"逐步学会"把概率集中到合理的下一个词上"。这也是第 11 章你会亲眼盯着看的曲线。

---

## 7.6 掩码损失:SFT 只对"回答"算账(第 12 章伏笔)

预训练时,语料是一整条 token 流,**每个位置都要预测下一个**,没有"该不该算"的区分。但到了 SFT(指令微调)阶段,一条训练样本长这样:

```
[系统/用户的 prompt 部分]  [assistant 的回答部分]
```

我们希望模型学会的是**怎么回答**,而**不是**去背诵、复现用户的提问。如果对 prompt 部分也算损失,模型会把一部分"学习力气"浪费在预测用户会问什么上——那不是我们要的。

解决办法是给每个位置配一个 0/1 的**掩码** $m_{b,t}$:assistant 回答的 token 处 $m=1$(要算损失),prompt 的 token 处 $m=0$(不算)。于是 SFT 损失变成"只在掩码为 1 的位置上求平均的交叉熵":

$$
\mathcal{L}_{\text{SFT}} = \frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}{\sum_{b,t} m_{b,t}}
$$

逐符号解释:$\ell_{b,t}$ 是位置 $(b,t)$ 的交叉熵;分子把每个位置的交叉熵乘上它的掩码再求和——掩码为 0 的位置(prompt)直接被乘没了,贡献为 0;分母 $\sum m_{b,t}$ 是"被算账的位置总数",用它做平均,保证结果是"**每个被监督的 token 的平均损失**",不受 prompt 长短影响。

对照 `src/post_training/sft.py` 的实现:

```python
mask = loss_mask[:, 1:].to(logits.dtype)

V = logits.size(-1)
ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1).long(), reduction="none")
ce = ce.view(targets.shape) * mask
return ce.sum() / mask.sum().clamp(min=1.0)
```

逐行解释:

- `mask = loss_mask[:, 1:]`:掩码同样要和标签一起**错位一位**(`[:, 1:]`),才能和 `targets` 对齐——这呼应 7.4.1 的 shift。
- `reduction="none"`:让 `cross_entropy` **不要**自动求平均,而是返回**每个位置各自**的交叉熵(一个向量),因为我们要先乘掩码再自己求平均。这正是上式里的 $\ell_{b,t}$。
- `ce = ce.view(targets.shape) * mask`:把逐位置交叉熵恢复成 $(B,T')$ 形状,逐元素乘掩码——对应分子里的 $m_{b,t}\,\ell_{b,t}$,prompt 位置被清零。
- `ce.sum() / mask.sum().clamp(min=1.0)`:分子求和、除以掩码之和(即被监督 token 数)——正是上式的分式。`clamp(min=1.0)` 是个保险:万一某批一个回答 token 都没有(分母为 0),避免除以零。

> **联系 \text{ignore\_index} 的常见做法**:很多代码库不用乘掩码,而是把不算损失的标签位置设成一个特殊值(常用 `-100`),再传给 `cross_entropy(..., ignore_index=-100)`,效果一样——那些位置被直接跳过。本仓库选择了"乘 0/1 掩码"这种更显式、更易读的写法。两种思路你都会在第 12 章细讲到。

---

## 7.7 动手:3 类小例子手算交叉熵,再用 torch 对照

我们用一个最小的 3 分类例子(把"3 类"想成"词表只有 3 个词")亲手算一遍交叉熵,再让 PyTorch 验证。假设模型在某个位置输出 logits

$$
z = [2.0,\; 1.0,\; 0.1]
$$

真实答案是第 0 类($y=0$)。**第一步**,算 softmax。先指数化:$\exp(2.0)\approx 7.389,\ \exp(1.0)\approx 2.718,\ \exp(0.1)\approx 1.105$;它们的和约为 $11.212$。于是

$$
p = \Big[\tfrac{7.389}{11.212},\ \tfrac{2.718}{11.212},\ \tfrac{1.105}{11.212}\Big] \approx [0.659,\ 0.242,\ 0.099]
$$

**第二步**,交叉熵就是 $-\log$ 真实类(第 0 类)的概率:

$$
\text{CE} = -\log(0.659) \approx 0.417
$$

**第三步**,验证一下梯度公式 $p-\text{onehot}(y)$:这里 $\text{onehot}(0)=[1,0,0]$,所以梯度约为 $[0.659-1,\ 0.242,\ 0.099]=[-0.341,\ 0.242,\ 0.099]$——第 0 类是负的(会被抬高),另两类是正的(会被压低),和 7.3.1 的结论一致。

现在用 `torch.nn.functional.cross_entropy` 对照(注意:**PyTorch 的 `cross_entropy` 直接吃 logits,内部自带 softmax+log,你不要自己先 softmax 再传进去**,否则等于做了两次):

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])   # 形状 (1, 3): 1 个样本, 3 个类
target = torch.tensor([0])                 # 真实类别是第 0 类

# 1) 手动: softmax -> 取真实类概率 -> -log
probs = F.softmax(logits, dim=-1)
print("softmax 概率:", probs)              # ≈ [[0.659, 0.242, 0.099]]
manual_ce = -torch.log(probs[0, 0])
print("手算交叉熵:", manual_ce.item())     # ≈ 0.417

# 2) 直接用 cross_entropy(吃 logits, 不要自己先 softmax)
ce = F.cross_entropy(logits, target)
print("F.cross_entropy:", ce.item())       # ≈ 0.417, 与手算一致

# 3) 顺便验证梯度 = p - onehot(y)
logits2 = logits.clone().requires_grad_(True)
F.cross_entropy(logits2, target).backward()
print("logits 的梯度:", logits2.grad)      # ≈ [[-0.341, 0.242, 0.099]]
```

运行后你会看到三件事互相印证:`F.cross_entropy` 的输出 ≈ 0.417,和你手算的 $-\log(0.659)$ 一致;`logits2.grad` ≈ `[-0.341, 0.242, 0.099]`,和 $p-\text{onehot}(y)$ 一致。把 `target` 改成 `torch.tensor([2])`(假装真实答案是模型最不看好的第 2 类),你会看到交叉熵一下子涨到约 $-\log(0.099)\approx 2.31$——模型对真实答案越意外,罚得越重,这就是 7.2.3 说的那股劲。

再算个困惑度感受一下:当 CE ≈ 0.417 时,这一步的困惑度是 $\exp(0.417)\approx 1.52$,意思是"模型在大约 1.5 个等可能选项之间纠结",已经相当笃定;而 CE ≈ 2.31 时困惑度是 $\exp(2.31)\approx 10.1$——在词表只有 3 个词的情况下,这已经比"瞎猜的 3"还差,说明模型把概率押错了地方。

---

## 小结

- 语言模型 = 一串条件概率连乘:$p_\theta(x_{1:T})=\prod_t p_\theta(x_t\mid x_{<t})$,把"建模整句话"化简成"给定前文预测下一个 token"。
- 训练目标由 **MLE → 取对数(连乘变连加)→ 取负** 推出**负对数似然 NLL**:"最大化似然 = 最小化 NLL"。
- 单个位置的 NLL 就是**交叉熵** $\text{CE}=-\log p_y=-\log\operatorname{softmax}(z)_y$;它对 logits 的**梯度是 $p-\text{onehot}(y)$**——干净、无饱和、推力正比于错误程度,这是交叉熵"好训"的根本。
- 整批损失是所有位置交叉熵的平均 $\mathcal{L}_{\text{LM}}=\frac{1}{BT}\sum\text{CE}$;输入与标签**错位一位**(输入 $t_0..t_{T-1}$、标签 $t_1..t_T$,代码里 `logits[:, :-1]` 配 `tokens[:, 1:]`)。
- **困惑度** $\text{PPL}=\exp(\mathcal{L})$:平均每步在多少个等可能选项里纠结;瞎猜模型 PPL ≈ $V$(本仓库约 5 万,对应初始 loss ≈ 10.83)。
- **掩码损失**:SFT 只对 assistant 回答的 token 算交叉熵(掩码为 1),prompt 部分掩码为 0 不计,损失对被监督 token 数求平均——为第 12 章铺路。

## 自测题

1. 为什么训练时要对似然取**对数**?它解决了什么数值问题、又带来了什么计算上的方便?
2. 写出交叉熵对 logits 的梯度公式,并用一句话解释:对"真实 token"对应的 logit,梯度是正还是负?它会被抬高还是压低?
3. 输入序列是 `[t0, t1, t2, t3]`,那么标签序列是什么?代码里用哪两个切片实现这个错位?
4. 某模型在某条 100 个 token 的句子上平均交叉熵是 $\ln 4$。它的困惑度是多少?这个数直觉上代表什么?
5. SFT 损失里,如果**不**乘掩码、对所有位置(含 prompt)都算交叉熵,会带来什么问题?

## 深入参考

- 本仓库精炼版参考:[目标、损失与困惑度](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/foundations/objectives_zh.md)
- 源码:`src/models/transformer.py`(`forward` 里的交叉熵)、`src/post_training/sft.py`(`sft_loss` 掩码损失)

下一章我们有了损失这个"方向盘",接下来要解决"怎么沿着它稳稳地走"——优化器与训练系统:从梯度下降到 Adam/AdamW、学习率调度、梯度累积与混合精度。

下一章 👉 [优化与训练系统](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-08-optimization)

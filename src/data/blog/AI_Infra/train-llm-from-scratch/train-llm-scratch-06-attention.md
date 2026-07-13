---
title: "从零训练大模型（六）：注意力机制完整推导"
author: Aidenz
pubDatetime: 2026-07-13T12:12:00Z
slug: train-llm-scratch-06-attention
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 6
tags:
  - LLM
  - 大模型
  - 从零训练
  - 注意力机制
description: "打开注意力这个黑箱：从缩放点积注意力、因果掩码，一路推到多头注意力，讲清它为什么是让 token 互相“通信”的引擎，以及复杂度从何而来。"
---

> **本章前置**:你已读过第 01–05 章。也就是说你已经知道:张量是带形状的多维数组、PyTorch 怎么做矩阵乘法和自动求导、文本如何被分词成 token 并打包成形状 $(B,T)$ 的整数批次、Transformer 的整体骨架(嵌入 → 若干个 Block → LayerNorm → LM head)以及"嵌入"是怎么把一个 token id 变成一个 $C$ 维向量的。
>
> **你将学到**:注意力(attention)到底在算什么、为什么这么算。我们会**一步不跳**地推导单头缩放点积注意力,弄清楚那个神秘的 $\sqrt{D}$ 从哪来;搞懂因果掩码(causal mask)如何让模型"看不到未来";理解为什么要拆成多个头(multi-head);最后亲手用一个 4-token 的小例子把带掩码的注意力权重算出来,看到漂亮的下三角。
>
> 👈 [上一章:解码器 Transformer 骨架](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-05-transformer) ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

## 6.1 直觉:每个 token 都要"环顾四周"

先抛开公式。想象你正在读这句话:

> 小明把**钥匙**放进口袋,然后他发现**它**不见了。

读到"它"的时候,你的大脑会自动回头去找:"它"指的是谁?是"钥匙"。你不是孤立地理解"它"这个字,而是**把它和前面相关的字联系起来**,从那些字里"取"信息过来,才明白这一整句话的意思。

这正是注意力要做的事。在 Transformer 里,经过嵌入之后,每个 token 都变成了一个 $C$ 维向量(在我们仓库里 $C$ 就是 `n_embed`)。但这个向量此刻还很"孤独"——它只携带了"我这个 token 自己"的信息,完全不知道上下文。注意力机制的任务,就是让每个位置的向量去**环顾前面所有的 token,按"相关程度"给它们打分,然后加权地把它们的信息汇总进来**,得到一个"读懂了上下文"的新向量。

我们把这件事拆成三个动作,后面会反复用到:

1. **我想找什么**(query,查询):站在当前位置,我提出一个"问题"。比如读到"它",我的问题大致是"前面哪个名词可能是我指代的对象?"
2. **我有什么**(key,键):每个位置都举一块"牌子",上面写着"我这里有什么样的信息",供别人来匹配。比如"钥匙"这个位置的牌子大致写着"我是个可被指代的物品名词"。
3. **匹配上了就把内容给你**(value,值):如果当前的问题(query)和某个位置的牌子(key)很匹配,就把那个位置真正要传递的内容(value)按匹配强度搬过来。

> 一个生活类比:你(query)拿着一张购物清单走进超市,每个货架都贴着标签(key),你拿清单去对每个标签,越对得上的货架,你越多地把那一格的商品(value)装进购物车。注意力就是"按相关度加权的信息购物"。

接下来我们把这三句大白话,一步步翻译成矩阵运算。

---

## 6.2 单头缩放点积注意力:从投影到加权求和

### 6.2.1 第一步:从隐藏状态投影出 Q、K、V

注意力的输入,是上一层(或嵌入层)输出的隐藏状态张量

$$
X \in \mathbb{R}^{B \times T \times C}
$$

逐符号解释:$B$ 是 batch(一批里有几条序列),$T$ 是序列长度(每条序列有几个 token),$C$ 是每个 token 的向量宽度(`n_embed`)。所以 $X$ 就是"这一批里,每条序列的每个 token,各自一个 $C$ 维向量"。

我们说过要为每个 token 造出三样东西:query、key、value。怎么造?**用三个可学习的线性投影**(就是三个不带偏置的矩阵乘法):

$$
Q = X W_Q,\qquad K = X W_K,\qquad V = X W_V
$$

逐符号解释:$W_Q, W_K, W_V$ 是三个**待训练的权重矩阵**,形状都是 $C \times D$。这里 $D$ 是"每个头的宽度"(head size)。$X$ 的最后一维是 $C$,乘上 $C \times D$ 的矩阵后,最后一维变成 $D$。于是

$$
Q,\,K,\,V \in \mathbb{R}^{B \times T \times D}
$$

也就是说:每个 token 现在有了三个 $D$ 维向量——它的 query 向量 $q$、key 向量 $k$、value 向量 $v$。"投影"听起来高深,其实就是"用一个学出来的矩阵,把 $C$ 维向量映射成 $D$ 维向量",目的是让模型自己学会"该提什么问题、该亮什么牌子、该传什么内容"。

在本仓库 `src/models/attention.py` 里,这三个投影就是三个 `nn.Linear`:

```python
self.key = nn.Linear(n_embed, head_size, bias=False)   # Key projection
self.query = nn.Linear(n_embed, head_size, bias=False) # Query projection
self.value = nn.Linear(n_embed, head_size, bias=False) # Value projection
```

`bias=False` 表示不加偏置项,纯粹是矩阵乘法 $XW$;`n_embed` 就是 $C$,`head_size` 就是 $D$。前向里调用它们:

```python
k = self.key(x)     # (B, T, head_size)
q = self.query(x)   # (B, T, head_size)
```

注释里的形状 `(B, T, head_size)` 正是我们说的 $B\times T\times D$。

### 6.2.2 第二步:用点积算"相关度分数"

现在每个 token 都有了 query 向量。位置 $t$ 想知道:"我该向位置 $s$ 借多少信息?"——衡量两个向量"合不合拍"的最简单办法,就是**点积**(dot product)。点积越大,说明两个向量方向越一致、越"对味"。

我们把位置 $t$ 的 query 向量 $q_t$ 和位置 $s$ 的 key 向量 $k_s$ 做点积,得到一个标量分数:

$$
S_{t,s} = q_t \cdot k_s = \sum_{d=1}^{D} q_{t,d}\, k_{s,d}
$$

逐符号解释:$q_{t,d}$ 是 query 向量 $q_t$ 的第 $d$ 个分量,$k_{s,d}$ 同理。把对应分量相乘再全部加起来,就是这个点积。$S_{t,s}$ 这个数越大,代表"位置 $t$ 觉得位置 $s$ 越相关"。

把所有 $(t,s)$ 对一次算完,就是一次矩阵乘法。对每条序列:

$$
S = Q K^\top
$$

逐符号解释:$Q$ 形状是 $T \times D$(暂时不看 batch 维),$K^\top$ 是 $K$ 的转置,形状 $D \times T$。两者相乘得到 $S$,形状 $T \times T$。$S$ 的第 $t$ 行第 $s$ 列,正好就是上面那个 $q_t \cdot k_s$。所以 $S$ 是一张"每个位置对每个位置的相关度打分表"。

加上 batch 维之后,$S \in \mathbb{R}^{B \times T \times T}$。**请记住这个 $T \times T$:它是注意力又强大又昂贵的根源,后面 6.4、6.5 节会反复回到它。**

### 6.2.3 第三步:为什么要除以 $\sqrt{D}$

如果直接拿 $S = QK^\top$ 去做后续的 softmax,会出问题。问题出在:**点积的数值大小会随着维度 $D$ 变大而变大**,而且是失控地变大。我们来给一个简单的方差论证。

假设 query 和 key 的每个分量都是相互独立、均值为 0、方差为 1 的随机数(这在训练初期、参数刚初始化时是合理的近似)。看单个点积

$$
S_{t,s} = \sum_{d=1}^{D} q_{t,d}\, k_{s,d}
$$

每一项 $q_{t,d} k_{s,d}$ 是两个独立、均值 0、方差 1 的随机数的乘积。它的均值是

$$
\mathbb{E}[q_{t,d} k_{s,d}] = \mathbb{E}[q_{t,d}]\,\mathbb{E}[k_{s,d}] = 0 \times 0 = 0
$$

它的方差是(对两个独立、零均值、单位方差的变量,乘积的方差等于各自方差之积):

$$
\operatorname{Var}(q_{t,d} k_{s,d}) = \mathbb{E}[q_{t,d}^2]\,\mathbb{E}[k_{s,d}^2] = 1 \times 1 = 1
$$

现在把 $D$ 个**互相独立**的这种项加起来。独立随机变量之和的方差等于各方差之和,于是

$$
\operatorname{Var}(S_{t,s}) = \sum_{d=1}^{D} 1 = D
$$

逐句解释这意味着什么:点积 $S_{t,s}$ 的方差是 $D$,对应的标准差是 $\sqrt{D}$。也就是说,**头越宽($D$ 越大),这些分数的取值范围就越大、越分散**。比如 $D=64$ 时,标准差约为 8,分数动辄就跑到 $\pm 16$ 这种量级。

为什么这很糟?因为下一步要对这些分数做 softmax。softmax 的特性是:当输入里有某个值明显比别人大,它就会把几乎全部概率(接近 1)压给那一个,其余压成接近 0。一旦分数被放大到 $\pm 16$ 这种尺度,softmax 输出会变成一个几乎是"非黑即白"(0/1)的尖锐分布。而对尖锐的 softmax 求梯度时,梯度几乎处处接近 0(因为输出已经饱和、对输入的变化不敏感了)。**梯度接近 0 = 学不动**。

解决办法非常直接:把方差缩回到 1。既然方差是 $D$,我们就把分数**除以 $\sqrt{D}$**:

$$
\operatorname{Var}\!\left(\frac{S_{t,s}}{\sqrt{D}}\right) = \frac{\operatorname{Var}(S_{t,s})}{(\sqrt{D})^2} = \frac{D}{D} = 1
$$

这样无论头宽 $D$ 是多少,缩放后的分数方差都稳定在 1 附近,softmax 不会一上来就饱和,梯度也就健康了。这就是"缩放点积注意力"里**缩放(scaled)**二字的来历。于是我们得到缩放后的分数:

$$
\tilde{S} = \frac{QK^\top}{\sqrt{D}}
$$

在代码里,这个缩放因子写成 `scale_factor`,在矩阵乘法时直接乘上去:

```python
scale_factor = 1 / math.sqrt(head_size)
# Calculate attention weights: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
attn_weights = q @ k.transpose(-2, -1) * scale_factor
```

`k.transpose(-2, -1)` 就是把 $K$ 的最后两维转置,实现 $K^\top$;`1 / math.sqrt(head_size)` 就是 $1/\sqrt{D}$。注释里的形状变化 `(B, T, head_size) @ (B, head_size, T) -> (B, T, T)` 正是我们推导的 $S \in \mathbb{R}^{B\times T\times T}$。

### 6.2.4 第四步:softmax 把分数变成权重

现在 $\tilde{S}$ 是一张分数表,但分数有正有负、不是概率。我们希望把位置 $t$ 对所有位置的分数,变成一组**加起来等于 1 的非负权重**(这样才能做"加权平均")。这正是 softmax 的工作。对 $\tilde{S}$ 的**每一行**单独做 softmax:

$$
A_{t,s} = \operatorname{softmax}_s(\tilde{S}_{t,:})_s = \frac{\exp(\tilde{S}_{t,s})}{\sum_{s'=1}^{T} \exp(\tilde{S}_{t,s'})}
$$

逐符号解释:分子 $\exp(\tilde{S}_{t,s})$ 把分数指数化(保证为正);分母把第 $t$ 行所有位置的指数加起来做归一化。结果 $A_{t,s}$ 就是"位置 $t$ 分给位置 $s$ 的注意力权重",它满足 $A_{t,s} \ge 0$ 且每一行 $\sum_s A_{t,s} = 1$。"按行 softmax"对应代码里的 `dim=-1`(在最后一维,也就是 $s$ 这一维上归一化):

```python
attn_weights = F.softmax(attn_weights, dim=-1)
```

### 6.2.5 第五步:用权重对 value 加权求和

最后一步,把权重作用到 value 上。位置 $t$ 的输出,就是用它那一行权重 $A_{t,:}$ 对所有位置的 value 向量做加权平均:

$$
\text{out}_t = \sum_{s=1}^{T} A_{t,s}\, v_s
$$

逐符号解释:$v_s$ 是位置 $s$ 的 value 向量($D$ 维)。我们用权重 $A_{t,s}$ 把它们加权加起来。某个位置越相关($A_{t,s}$ 越大),它的 value 就越多地进入位置 $t$ 的输出。这正是 6.1 节那个"按相关度加权的信息购物"的精确版本。

写成矩阵就是 $A$ 乘 $V$,把整个单头注意力合起来:

$$
\boxed{\;\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D}}\right)V\;}
$$

形状追踪:$A \in \mathbb{R}^{B\times T\times T}$,$V \in \mathbb{R}^{B\times T\times D}$,相乘得到**输出形状** $\mathbb{R}^{B\times T\times D}$——和进来的 query/value 一样宽。换句话说,注意力是一个"形状不变"的信息混合器:进去 $B\times T\times D$,出来还是 $B\times T\times D$,只是每个位置的向量已经"读过了上下文"。代码里:

```python
v = self.value(x)   # (B, T, head_size)
# Apply attention weights to values
out = attn_weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
```

注释里的 `(B, T, T) @ (B, T, head_size) -> (B, T, head_size)` 就是 $A V$ 这一步。

---

## 6.3 因果掩码:不许偷看未来

到这里有一个**致命问题**还没处理。我们的模型是**自回归**(autoregressive)的:训练时,它在每个位置 $t$ 都要预测"下一个 token 是什么"。也就是说,在位置 $t$,模型只被允许看到 $0, 1, \ldots, t$ 这些位置(包括自己),**绝对不能看到 $t+1$ 及之后**——因为 $t+1$ 那个 token 恰恰就是它要预测的答案!如果让它偷看,它会直接"抄答案",训练出来的损失漂亮得不真实,但一到真正生成(此时未来根本不存在)就彻底废掉。

可是我们刚算出来的分数表 $\tilde{S}$ 是**满的** $T\times T$:位置 $t$ 对位置 $t+1, t+2, \ldots$ 都打了分。我们需要把这些"看向未来"的分数**屏蔽掉**。

办法很巧妙:在 softmax **之前**,把所有"未来位置"的分数改成 $-\infty$。回忆 softmax 里有 $\exp(\cdot)$,而

$$
\exp(-\infty) = 0
$$

所以这些位置经过 softmax 后,权重恰好变成 0——等于"看不见"。具体来说,我们只保留 $s \le t$ 的位置:

$$
\tilde{S}^{\text{masked}}_{t,s} =
\begin{cases}
\tilde{S}_{t,s}, & s \le t \quad(\text{允许:当前或过去}) \\
-\infty, & s > t \quad(\text{屏蔽:未来})
\end{cases}
$$

"允许的位置"恰好构成一个**下三角**形状(对角线及以下)。以 $T=5$ 为例,允许矩阵长这样(1 表示能看,0 表示屏蔽):

$$
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 & 1
\end{bmatrix}
$$

读法:第 1 行(位置 0)只能看自己;第 2 行(位置 1)能看位置 0 和 1;……第 5 行能看全部 5 个。每个位置都"只能往回看"。

本仓库的实现非常贴合上面这套逻辑。它在 `__init__` 里用 `torch.tril`(取下三角)预先造好一个常量矩阵,并用 `register_buffer` 存起来(buffer 是"不参与训练、但跟着模型走"的张量):

```python
# Lower triangular matrix for causal masking
self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
```

`torch.ones(context_length, context_length)` 造一个全 1 的方阵,`torch.tril` 把上三角(未来部分)清零,留下我们要的下三角。然后在 `forward` 里,用 `masked_fill` 把 `tril` 中为 0 的位置(也就是未来)填成 `-inf`:

```python
# Apply causal masking
attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
attn_weights = F.softmax(attn_weights, dim=-1)
```

`self.tril[:T, :T]` 是按当前实际序列长度 $T$ 切出左上角(因为缓冲区是按最大 `context_length` 造的,而当前 batch 的 $T$ 可能更短);`== 0` 找出所有未来位置;`masked_fill(..., float('-inf'))` 把它们灌成 $-\infty$。紧接着的 `F.softmax` 就把这些 $-\infty$ 变成了 0 权重。顺序很关键:**先填 $-\infty$,再 softmax**——反了就不对了。

---

## 6.4 多头注意力:多个视角并行看

### 6.4.1 为什么要"多个头"

一个注意力头,本质上学到的是**一种**"相关度模式"。但语言里同时存在很多种不同的依赖关系需要追踪:

- 代词指代谁(语义指代);
- 主谓是否一致(语法);
- 相邻几个字组成的短语(局部结构);
- 引号、括号、分隔符的配对(格式跟踪);
- 类似代码/算术里的对应关系。

指望**一个**头同时把这些都学好,太勉强了。于是我们干脆放 $H$ 个头(`n_head`)**并行**地各看各的:让它们在各自的子空间里,自由地学不同的模式。这就像看同一份文件时,派出好几位专家——一位盯语法、一位盯指代、一位盯格式——最后再把各自的发现汇总。

### 6.4.2 怎么拆、怎么拼

做法很直接:把宽度 $C$ 平分给 $H$ 个头,每个头分到

$$
D = \frac{C}{H}
$$

逐符号解释:$D$ 是每个头的宽度(head size),等于总宽度 $C$ 除以头数 $H$。每个头都独立地走一遍 6.2–6.3 节那套完整的(带因果掩码的)单头注意力,各自输出一个 $B\times T\times D$ 的张量。

然后把 $H$ 个头的输出沿最后一维**拼接**(concatenate)起来。因为每个头宽 $D$、共 $H$ 个,拼完正好是 $H \times D = C$:

$$
\text{Concat}(\text{head}_1, \ldots, \text{head}_H) \in \mathbb{R}^{B\times T\times C}
$$

最后再过一个线性投影 `proj`,让各个头之间的信息互相**混合**一下(否则各头的输出只是简单并排,没交流):

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W_O
$$

逐符号解释:$W_O$ 是输出投影矩阵(形状 $C\times C$),它把拼接结果重新搅拌成最终输出,形状仍是 $B\times T\times C$。

对照本仓库 `MultiHeadAttention`:构造时建了一个 `ModuleList`,装 `n_head` 个 `Head`,每个头的宽度正是 `n_embed // n_head`(也就是 $C/H$);再加一个输出投影 `proj`:

```python
self.heads = nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])
self.proj = nn.Linear(n_embed, n_embed)
```

前向时,列表推导式 `[h(x) for h in self.heads]` 让每个头各算各的,`torch.cat(..., dim=-1)` 沿最后一维拼接,最后 `self.proj` 做输出投影:

```python
# Concatenate the output of each head along the last dimension (C)
x = torch.cat([h(x) for h in self.heads], dim=-1)
# Apply final linear projection
x = self.proj(x)
```

> **一个实现细节**:这份教学代码为了好读,是用一个 Python `for` 列表**逐个头**串行计算的。工业级实现通常会把它向量化成一个 $(B, H, T, D)$ 的大张量一次算完,效率更高,但数学完全等价。理解上,你只要记住"$H$ 个独立的单头,各算各的,再拼起来"即可。

### 6.4.3 注意力张量与显存:为什么是 $(B, H, T, T)$

把多头一起看,核心的注意力分数张量(每个头一张 $T\times T$ 的表)合在一起的形状是

$$
(B,\; H,\; T,\; T)
$$

逐符号解释:$B$ 条序列、每条 $H$ 个头、每个头一张 $T\times T$ 的权重表。注意最后那两个 $T$:**这张表的大小正比于 $T^2$**。这件事直接决定了显存开销——上下文越长,这个张量越大,而且是**平方级**地变大。这是第 6.5 节"复杂度"的实物来源。

---

## 6.5 复杂度:$O(T^2 D)$ 与长上下文的代价

我们数一数单个头里最重的两步运算量(用大 O 记号,只看量级):

- **算分数** $QK^\top$:$Q$ 是 $T\times D$,$K^\top$ 是 $D\times T$,相乘得到 $T\times T$,每个元素是 $D$ 次乘加。总共约 $T \times T \times D = T^2 D$ 次运算。
- **加权求和** $AV$:$A$ 是 $T\times T$,$V$ 是 $T\times D$,同样约 $T^2 D$ 次运算。

所以单头的时间复杂度是

$$
O(T^2 D)
$$

而存储那张注意力表的空间复杂度(每头)是 $O(T^2)$,把 batch 和多头算上就是 $O(B H T^2)$。

逐句解读这意味着什么:**计算量和显存都随序列长度 $T$ 平方增长**。把上下文长度翻一倍($T \to 2T$),注意力部分的计算和那张表的内存就要变成原来的 4 倍($2^2$)。这就是"长上下文很贵"的根本原因,也是后来各种高效注意力(稀疏注意力、FlashAttention、滑窗等)拼命想优化的目标。本仓库为了教学清晰,**故意**把这张 $T\times T$ 表显式地"物化"出来(就是你在代码里看到的 `attn_weights`),不做任何省内存的花招,方便你看清每一步。

---

## 6.6 动手:用一个 4-token 小例子手算带掩码的注意力

光看公式不踏实,我们用 PyTorch 跑一个能用肉眼核对的迷你例子:1 条序列、4 个 token($B=1, T=4$),头宽 $D=2$。重点不是数值多精确,而是**亲眼看到下三角和"每行权重和为 1"**。把下面代码存成文件运行,或在 Python 交互环境里逐行敲:

```python
import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

B, T, D = 1, 4, 2          # 1 条序列, 4 个 token, 每个头宽度 D=2
q = torch.randn(B, T, D)   # 假装这是 query 投影的结果
k = torch.randn(B, T, D)   # key
v = torch.randn(B, T, D)   # value

# 第一步: 算缩放后的分数 (B, T, T)
scale = 1 / math.sqrt(D)
scores = q @ k.transpose(-2, -1) * scale
print("缩放后的分数 scores:\n", scores)

# 第二步: 造下三角掩码, 把未来位置填成 -inf
tril = torch.tril(torch.ones(T, T))
print("下三角 tril:\n", tril)
masked = scores.masked_fill(tril == 0, float('-inf'))
print("掩码后的分数:\n", masked)

# 第三步: 按行 softmax 得到权重
weights = F.softmax(masked, dim=-1)
print("注意力权重 weights:\n", weights)
print("每行求和(应该都≈1):", weights.sum(dim=-1))

# 第四步: 加权求和得到输出 (B, T, D)
out = weights @ v
print("输出 out 形状:", out.shape)
```

你会观察到几件关键的事:

1. `tril` 打印出来正是那个由 1 和 0 组成的下三角矩阵。
2. `masked` 里,右上三角(未来位置)全变成了 `-inf`。
3. `weights` 里,**右上三角全部是 0**(因为 $\exp(-\infty)=0$),而且它本身也是个下三角概率矩阵。比如第 1 行只有第 1 个位置非零且等于 1(位置 0 只能看自己,只能把全部注意力给自己);第 2 行只有前两个位置非零、加起来为 1;依此类推。
4. `每行求和` 打印出来全是约等于 1 的数——印证了"每个位置把它的注意力按概率分配出去,总量是 1"。
5. `out` 的形状是 `(1, 4, 2)`,和输入 query/value 一样宽——印证了 6.2.5 节说的"形状不变的信息混合器"。

如果你想把它和仓库代码对齐,可以直接 `import` 真实模块跑一遍(注意 `Head` 内部会自己做 $Q,K,V$ 投影,所以只要喂随机隐藏状态 `x` 即可):

```python
from src.models.attention import Head
import torch

x = torch.randn(1, 4, 8)                 # (B=1, T=4, C=8)
head = Head(head_size=4, n_embed=8, context_length=4)
out = head(x)
print(out.shape)                          # torch.Size([1, 4, 4]) -> (B, T, head_size)
```

动手改一改、加深印象:把 `T` 从 4 改成 6,看下三角变大;把 `scale` 那一行改成 `scale = 1`(去掉缩放),再把 `q`、`k` 乘以一个较大的数(比如 `q = torch.randn(B, T, D) * 10`),你会看到 softmax 后的权重变得非常"非黑即白"(某个位置接近 1、其余接近 0)——这正是 6.2.3 节预言的"分数太大导致 softmax 饱和"的现象,你就明白为什么要除以 $\sqrt{D}$ 了。

---

## 小结

- 注意力让每个 token **环顾前面相关的 token,按相关度加权地汇总它们的信息**。三个角色:query(我想找什么)、key(我有什么)、value(匹配上就传什么)。
- 单头缩放点积注意力:$\text{Attention}(Q,K,V)=\operatorname{softmax}\!\big(\frac{QK^\top}{\sqrt{D}}\big)V$。其中 $Q=XW_Q$ 等三个线性投影;分数 $S=QK^\top$ 是 $T\times T$ 的相关度表;**除以 $\sqrt{D}$** 是为了把点积方差从 $D$ 缩回 1,避免 softmax 饱和、梯度消失。
- 因果掩码:自回归不能偷看未来,所以在 softmax **之前**把未来位置的分数填成 $-\infty$(代码里用下三角 `tril` 缓冲区 + `masked_fill`),softmax 后这些位置权重恰好为 0,形成下三角的注意力。
- 多头注意力:把宽度 $C$ 平分给 $H$ 个头(每头 $D=C/H$),各头并行学不同模式,输出拼接回 $C$ 维再过输出投影混合。核心张量形状 $(B,H,T,T)$,显存随 $T^2$ 增长。
- 复杂度 $O(T^2 D)$:上下文翻倍,注意力计算与内存翻 4 倍——这就是长上下文昂贵的根源。

## 自测题

1. 为什么要把 $QK^\top$ 除以 $\sqrt{D}$ 而不是别的数(比如 $D$ 或 $\sqrt{C}$)?用"方差"两个字解释清楚。
2. 因果掩码为什么要在 softmax **之前**填 $-\infty$,而不是在 softmax **之后**把未来位置的权重直接置 0?(提示:想想 softmax 之后每行还能不能保证和为 1。)
3. 如果 `n_embed = 64`、`n_head = 8`,那么每个头的宽度 $D$ 是多少?多头拼接之后的宽度是多少?
4. 注意力分数张量的形状是 $(B, H, T, T)$。如果你把 batch 大小 $B$ 翻倍、序列长度 $T$ 翻倍,这个张量占的内存大约变成原来的几倍?
5. 在 6.6 的动手代码里,为什么 `weights` 的第 1 行(位置 0)一定是 `[1, 0, 0, 0]`?

## 深入参考

- 本仓库精炼版参考:[注意力、掩码与多头](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/foundations/attention_zh.md)
- 源码:`src/models/attention.py`(`Head`、`MultiHeadAttention`)、`src/models/transformer_block.py`(注意力如何嵌进残差结构)

下一章我们要回答另一个核心问题:有了这些隐藏状态和 logits,模型到底**根据什么信号去学习**?也就是训练目标——交叉熵与困惑度的完整推导。

下一章 👉 [训练目标:交叉熵与困惑度 · 完整推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-07-objectives)

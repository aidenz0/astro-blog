---
title: "从零训练大模型（九）：生成与采样"
author: Aidenz
pubDatetime: 2026-07-13T08:09:00Z
slug: train-llm-scratch-09-generation
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 9
tags:
  - LLM
  - 大模型
  - 从零训练
  - 解码策略
description: "自回归生成的完整机制，以及贪心、温度、top-k、top-p 等采样策略的原理与取舍——同一个模型，如何生成出风格迥异的文本。"
---

> **本章前置**:第 04 章(token 与 token id)、第 05 章(Transformer 输出 logits)、第 06 章(因果掩码:每个位置只能看左边)、第 07 章(softmax 把 logits 变成下一个 token 的概率分布)。
>
> **你将学到**:模型训好之后,怎么让它一个字一个字地"写"出文本——**自回归生成主循环**;从 logits 选下一个 token 的几种策略及其权衡——**贪心、温度采样、top-k、top-p(nucleus)**,每种都给公式;**结束 token** 和上下文长度上限怎么决定生成何时停;为什么生成和训练**共用同一套核心代码**;以及采样参数如何在"创造性 vs 准确性"之间拨动旋钮。最后带你用 `scripts/chat.py` 把这些 flag 真正对比一遍。
>
> 👈 [上一章:优化与训练系统](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-08-optimization) ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

第 07 章里,模型对一段文本做一次前向,在**每个位置**都输出一份 logits,告诉我们"这个位置后面接什么 token"。训练时,这些"后面接什么"的答案我们早就知道(它就写在数据里),所以叫 **teacher forcing(老师强制喂入正确答案)**。

但**生成**时没有老师了。模型得自己"写":写出一个 token,把它接到句子末尾,再读着自己刚写的去预测下一个……如此循环。这一章讲的就是这个"自己喂自己"的循环,以及"每一步到底怎么挑下一个 token"。

## 一、自回归生成主循环

"自回归(autoregressive)"的意思是:**下一步的输入,包含上一步自己的输出**。生成主循环就五步,转一圈吐一个 token:

1. 把"目前已生成的整个序列"喂进模型,做一次前向;
2. 只取**最后一个位置**的 logits(因为有因果掩码,这个位置已经"看过"了前面全部上下文,正好用来预测下一个);
3. 用某种策略从这份 logits 里**选出一个 token**(下面几节就在讲这一步);
4. 把选出的 token **拼接**到序列末尾;
5. 回到第 1 步,直到遇到**结束 token**或达到**最大长度**。

本仓库里最朴素的一版实现在 `src/models/transformer.py`(教学用):

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.context_length:]
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
```

逐行对照上面五步:

- `idx[:, -self.context_length:]`:只保留最后 `context_length` 个 token 喂进去(超出窗口的旧 token 丢掉,详见第三节);
- `logits, _ = self(idx_cond)`:前向。返回的第二项是损失,生成时不需要,用 `_` 丢弃;
- `logits[:, -1, :]`:**取最后一个位置**的 logits,形状从 `(B, T, vocab)` 变成 `(B, vocab)`;
- `F.softmax(...)` + `torch.multinomial(...)`:把 logits 变成概率,再按概率抽一个 token(这一版直接从完整分布里采样,没有任何截断);
- `torch.cat(...)`:把新 token 接到序列末尾,进入下一轮。

> **为什么每次都重新喂入整个序列、却只用最后一个位置?** 因为有了因果掩码(第 06 章),序列里第 $t$ 个位置的输出,本就只依赖第 $0\dots t$ 个 token。我们要预测的是"已生成序列之后的那个新 token",它对应的正是**最后一个位置**的输出。前面那些位置的 logits 在生成时用不上(它们预测的是序列内部已经存在的 token)。这版实现为清晰起见每步都把前缀重算一遍;真正追求速度的实现会用 KV cache 把前面算过的中间结果缓存下来——本仓库的 `src/post_training/rollout.py` 注释明确说明它**故意不用 KV cache**(`No KV cache (kept for clarity)`),因为这里的序列短,清晰比那点速度更重要。

## 二、从 logits 到 token:四种策略

第 1、2、4、5 步都是固定的;真正决定"模型说话风格"的是第 3 步——**怎么从 logits 选 token**。本仓库后训练的采样逻辑集中在 `src/post_training/rollout.py` 的 `filter_logits` 函数里,我们对着它讲四种策略。

### 1. 贪心解码(greedy / argmax)

最简单:每步都选**概率最高**的那个 token。

$$
x_t = \arg\max_i\; p_i
$$

逐符号:$p_i$ 是词表里第 $i$ 个 token 的概率,$\arg\max$ 取使它最大的那个 $i$。

- **优点**:确定性(同样的输入永远得到同样的输出),适合做评估、做对比——本仓库 `batched_generate(..., greedy=True)` 正是用它跑 GSM8K 评估,这样不同阶段的模型能在同一把尺子下比较;
- **缺点**:呆板、容易**重复**("the the the…"),也容易陷进一条"局部看着最优、整体很无聊"的路径。

代码里 `greedy=True` 会被翻译成 `temperature, top_k, top_p = 1.0, 1, None`——也就是"只保留 top-1",等价于 argmax(见 `batched_generate`)。

### 2. 温度采样(temperature)

不总挑最高的,而是**按概率随机抽**(概率高的更可能被抽到,但低概率 token 也有机会)。**温度(temperature)**$T$ 是一个旋钮,用来在 softmax 之前给 logits 整体缩放,从而调节这个分布的"尖锐 / 平坦"程度:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

逐符号:$z_i$ 是第 $i$ 个 token 的 logit,$T$ 是温度。关键看 $z_i/T$ 这个除法怎么改变分布:

- $T < 1$(如 0.5):相当于把所有 logit **放大**,差距被拉开,softmax 后分布更**尖锐**——高概率 token 更碾压,输出更**确定、稳妥**,但多样性低;
- $T = 1$:不变(就是原始 softmax);
- $T > 1$(如 1.5):把 logit **压缩**,差距变小,分布更**平坦**——低概率 token 也更容易被抽中,输出更**随机、有创造性**,但也更容易跑偏、出错;
- $T \to 0$:分布无限尖锐,退化成贪心(永远抽最高的那个)。

记一句话:**温度作用在 logits 上,不是在概率上**;低温更确定,高温更狂野。代码里就一行:

```python
if temperature != 1.0:
    logits = logits / max(temperature, 1e-6)
```

(`max(temperature, 1e-6)` 只是防止有人传 0 进来除爆。)

### 3. Top-k 采样

光靠温度还不够:就算分布被温度压平了,词表里几万个 token 的**长尾**加起来仍有不小概率,偶尔会抽到一个完全不着边的词。**top-k** 的办法是先**只保留概率最高的 $k$ 个 token**,其余全部丢弃(概率置 0),再在这 $k$ 个里按概率采样。

$$
\text{候选集} = \{\,\text{logits 最大的前 } k \text{ 个 token}\,\}, \qquad \text{其余 logit} \leftarrow -\infty
$$

把不要的 logit 设成 $-\infty$,softmax 后它们的概率自然为 0。代码:

```python
if top_k is not None and top_k > 0:
    k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(logits < kth, float("-inf"))
```

它先找出"第 $k$ 大的 logit 值"`kth`,然后把所有小于它的 logit 填成 `-inf`。

- **权衡**:$k$ 小 → 候选窄、更稳但更单调;$k$ 大 → 候选宽、更多样但更可能采到劣质 token。
- **局限**:$k$ 是个固定数字,不看分布形状。可模型有时非常确定(top-1 就占 99% 概率),有时很犹豫(概率摊得很平)。固定的 $k$ 在前一种情况下嫌多,在后一种情况下嫌少。这正是 top-p 要解决的。

### 4. Top-p / 核采样(nucleus sampling)

**top-p** 换了个思路:不固定"留几个",而是固定"留多少概率质量"。把 token 按概率从高到低排好,从最高的开始累加,**一旦累计概率达到阈值 $p$ 就停**,保留到此为止的这一小撮 token(称为 "nucleus,核"),其余丢弃。

$$
\text{保留最小的集合 } S,\ \text{使得} \sum_{i \in S} p_i \ge p
$$

它的好处是**候选集大小随模型的确定程度自动伸缩**:模型很确定时(某个 token 概率就 0.95),一两个 token 就凑够 $p$,候选集自动变窄;模型很犹豫时(概率摊得平),要凑够 $p$ 就得纳入更多 token,候选集自动变宽。这通常比固定 $k$ 更自然。代码:

```python
if top_p is not None and 0.0 < top_p < 1.0:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumprobs > top_p
    # Keep at least the top token; shift so the token that crosses top_p stays.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = remove.scatter(-1, sorted_idx, remove)
    logits = logits.masked_fill(remove, float("-inf"))
```

读它的逻辑:先把 logits 降序排,算累计概率 `cumprobs`;`cumprobs > top_p` 标记出"越过阈值之后"的 token;那两行 `remove[..., 1:] = remove[..., :-1]` 加 `remove[..., 0] = False` 是一个小心的"右移一位",保证**那个刚好把累计概率推过 $p$ 的 token 本身被保留**,而且**至少留住概率最高的 token**(避免极端情况下把所有候选都删光);最后 `scatter` 把标记还原回原始词表顺序,再用 `-inf` 抹掉。

**这几种策略可以叠加**:典型用法是先温度、再 top-k 和/或 top-p。`filter_logits` 正是按"温度 → top-k → top-p"的顺序依次施加,最后返回处理好的 logits 交给 softmax 采样。本仓库 `scripts/chat.py` 的默认值就是温度 0.8 配 top-p 0.95——温和地保留主流候选,既不死板也不胡来。

## 三、什么时候停:结束 token 与最大长度

生成不能无限转下去,两个东西决定它何时停:

**① 结束 token(EOT)。** 分词器有一个特殊 token `<|endoftext|>`,本仓库里它的 id 是 **50256**(`src/post_training/chat_template.py` 里的 `EOT_ID`)。预训练时,它被插在文档与文档之间、以及助手回答结束之后,模型于是**学会**"该收尾时就吐出这个 token"。生成时,一旦采到 EOT 就停。`rollout.py` 里把它作为默认停止符:

```python
stop_tokens: tuple[int, ...] = (EOT_ID,),
```

每一行序列各自独立判停:某行采到 EOT 后就标记 `finished`,之后的位置填充 pad、不再计入有效输出。解码时还会把 EOT 之后的内容截掉(`batched_generate` 里 `toks[: toks.index(EOT_ID)]`)。

> 这也解释了一个常见现象:如果一个模型**没被训练**好好输出 EOT(比如纯 base 模型、或 SFT 不充分),它就不知道何时停,只能一直写到撞上长度上限——这正是第 12 章 SFT 要教会模型的事情之一。

**② 最大长度上限。** 模型的上下文窗口是固定的 `context_length`,它同时管着"能读多长"和"prompt+生成总共能有多长"。本仓库强约束:

$$
\text{prompt\_len} + \text{max\_new\_tokens} \le \text{context\_length}
$$

`rollout.py` 里就是这么算并夹住生成预算的:

```python
max_new_tokens = min(max_new_tokens, cap - P)
if max_new_tokens <= 0:
    raise ValueError(f"Prompt length {P} leaves no room under context_length {cap}.")
```

逐符号:`P` 是 prompt 长度,`cap` 是 `context_length`。如果 prompt 本身就快把窗口填满,留给生成的空间(`cap - P`)就所剩无几,甚至报错。

而当对话越来越长、超过窗口时,主循环靠 `idx[:, -context_length:]` **只保留最后 `context_length` 个 token**,最旧的会被丢弃——模型再也"看不到"被挤出窗口的内容。所以上下文长度不只是个训练超参,它直接决定了产品层面"模型能记住多长的对话",是一条硬约束。

## 四、生成与训练为什么共用同一套核心

你可能注意到了:`src/post_training/rollout.py` 这个文件,既被推理用(`inference.py` → `chat.py`),又被强化学习训练用(PPO / GRPO)。这不是偶然,而是**刻意**的设计。

文件顶部注释把原因写得很清楚:同样的 log-prob 计算,要在 **PPO/GRPO 中对四套不同的参数**反复跑——可训练的策略、冻结的参考模型、旧策略快照、以及带 value head 的 actor-critic 包装。把它们写成自由函数 `f(model, ...)` 而不是绑在模型上的方法,能在这四种场景间自由组合复用。

更深一层的道理是:**训练时算的"某个 token 的概率",必须和生成时实际采样所依据的概率严格一致**,否则强化学习的梯度就对不上号。所以——

- `generate_with_logprobs`:边采样,边记下每个生成 token 的 log-prob(采样时用的就是这个分布);
- `compute_logprobs`:训练时用 teacher forcing **重算**同一批序列的 log-prob,而且严格复刻模型训练时的"错位"对齐(`logits[:, t]` 预测 `sequences[:, t+1]`,见第 07 章的标签错位);
- 两者都把温度、log-softmax 的算法对齐,连前面第 08 章提到的"log-prob 一律转 fp32"也是同一处保证。

让生成和训练共享这套核心,就是为了保证这种**一致性**。这一点会在第 15 章(PPO)、第 16 章(GRPO)真正发挥作用,这里先埋下伏笔。

## 五、采样如何拨动"创造性 vs 准确性"

把前面的参数串起来,你就有了一组调"性格"的旋钮:

| 想要 | 怎么调 | 代价 |
|---|---|---|
| 最稳、可复现(评估、数学题) | `--greedy`,或低温(T≈0.2) + 小 top-k | 单调、易重复 |
| 平衡(日常对话) | T≈0.7–0.8 + top-p≈0.9–0.95 | —(本仓库默认) |
| 更发散、有创意(头脑风暴) | 高温(T≈1.0–1.2) + 大 top-p | 更易跑题、出错 |

核心权衡只有一句:**越确定 → 越准但越无聊;越随机 → 越有创意但越容易胡说**。做需要正确答案的任务(算术、代码、事实问答)就往"确定"端拨;做需要花样的任务(写故事、起名字)就往"随机"端拨。

为什么高温更容易"胡说"?回忆第 07 章那条反馈链:生成是自回归的,一旦在早期采到一个糟糕的 token,后面所有预测都建立在这个错误前缀之上,错误会被**放大**。高温提高了采到坏 token 的概率,于是更容易把自己带进沟里。这也是后训练(SFT / 偏好对齐 / RLVR)要解决的核心问题之一——让模型即便在采样下也更可靠。

## 六、动手:对比不同采样设置

我们用 `scripts/chat.py` 把上面的旋钮真正拨一拨。先看它支持的 flag(都来自源码,真实存在):

| flag | 含义 | 默认 |
|---|---|---|
| `--ckpt` | checkpoint 路径(必填) | — |
| `--prompt` | 一次性输入;省略则进入交互式 REPL | 无(进 REPL) |
| `--raw` | base 模型续写模式(不套对话模板) | 关 |
| `--greedy` | 确定性 argmax 解码 | 关 |
| `--temperature` | 温度 $T$ | 0.8 |
| `--top_p` | 核采样阈值 $p$ | 0.95 |
| `--top_k` | top-k 的 $k$ | 无 |
| `--max_new_tokens` | 最多生成多少 token | 256 |
| `--system` | 可选的 system 提示(对话模式) | 无 |

> **重要前提**:你现在还没有任何 checkpoint(`<某 checkpoint>` 只是占位)。本节是"概念演示"——先看清命令长什么样、各参数怎么搭配。等学完**第 11 章**亲手预训练出 `base_pretrained.pt`(或后续阶段的 `sft.pt` / `grpo.pt`)之后,**回到这里把路径换成真实文件**,就能跑出实际对比了。

对同一个 prompt 换不同采样设置,直观感受"性格"变化:

```bash
# 1) 贪心:确定性,跑几次输出完全一样,稳但可能呆板
PYTHONPATH=. python scripts/chat.py --ckpt <某 checkpoint> \
    --prompt "Write one sentence about the sea." --greedy

# 2) 低温:仍偏稳妥,但有一点随机性
PYTHONPATH=. python scripts/chat.py --ckpt <某 checkpoint> \
    --prompt "Write one sentence about the sea." --temperature 0.3

# 3) 默认(温度 0.8 + top-p 0.95):平衡,日常对话推荐
PYTHONPATH=. python scripts/chat.py --ckpt <某 checkpoint> \
    --prompt "Write one sentence about the sea."

# 4) 高温 + 大 top-p:更发散,跑几次差异明显,但更易跑题
PYTHONPATH=. python scripts/chat.py --ckpt <某 checkpoint> \
    --prompt "Write one sentence about the sea." --temperature 1.2 --top_p 0.98

# 5) top-k 限制候选数量(只在概率最高的 40 个里采)
PYTHONPATH=. python scripts/chat.py --ckpt <某 checkpoint> \
    --prompt "Write one sentence about the sea." --top_k 40
```

观察要点:
- 把 **1)** 跑两遍,输出一模一样(确定性);把 **4)** 跑两遍,输出明显不同(随机性);
- 从 **2) → 3) → 4)** 温度递增,你会感到输出从"中规中矩"逐渐变"花哨/不稳定";
- 用 **base 模型**(配 `--raw`)时它只会"续写"而不会"回答指令",这恰好印证了为什么需要 SFT(第 12 章)——这也是一个值得回来亲手验证的对比。

> 顺带一提:`scripts/chat.py` 里有个小细节——当 `--top_p` 取到 1(或更大)时,代码会把它当作"不启用 top-p"(`top_p=args.top_p if args.top_p < 1 else None`),因为 $p=1$ 等于保留全部 token、相当于没截断。

## 小结

- **自回归生成**就是"喂入已生成序列 → 取最后位置 logits → 选下一个 token → 拼接 → 重复",直到 EOT 或撞上长度上限;
- 从 logits 选 token 有四把工具:**贪心**(最稳)、**温度**(调尖锐/平坦)、**top-k**(留前 k 个)、**top-p**(留够 p 的概率质量,候选随分布自适应),可叠加使用;
- 何时停由 **EOT(`<|endoftext|>`, id=50256)** 和 **`prompt+生成 ≤ context_length`** 共同决定;
- 生成与训练**共用 `rollout.py` 的核心**,是为了保证"采样所依据的概率"和"训练时重算的概率"严格一致,这对 PPO/GRPO 至关重要;
- 采样参数本质是在**创造性与准确性**之间拨旋钮:越确定越准越闷,越随机越活越易错。

## 自测题

1. 自回归生成主循环的五个步骤是什么?为什么每步只取**最后一个位置**的 logits?
2. 温度 $T$ 作用在 logits 还是概率上?$T=0.5$、$T=1$、$T=2$、$T\to 0$ 分别让分布和输出怎么变?
3. top-k 和 top-p 都在"截断候选集",它们的关键区别是什么?为什么 top-p 常被认为更自然?
4. `--greedy` 在本仓库内部被翻译成了哪组 `(temperature, top_k, top_p)`?为什么它等价于 argmax?
5. EOT token 的 id 是多少?如果一个模型从没学会输出 EOT,生成时会发生什么?这暗示了 SFT 的什么作用?
6. 已知 `context_length=1024`,prompt 占了 1000 个 token,你设 `max_new_tokens=256`,实际最多能生成几个 token?
7. 为什么本仓库要让生成(`generate_with_logprobs`)和训练重算(`compute_logprobs`)共用同一套 log-prob 逻辑?这和第 08 章"log-prob 转 fp32"是同一类考量吗?
8. 你要让模型解一道数学题(要正确答案),又要让它写一首即兴小诗(要花样),两种任务你各自会怎么设温度 / top-p?为什么?

## 深入参考

- 工程速查:[`docs/zh/foundations/generation_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/foundations/generation_zh.md)(含"生成跑偏诊断表":重复 / 忽略指令 / 格式错乱各查什么)
- 源码:`src/post_training/rollout.py`(`filter_logits` / `generate_with_logprobs` / `compute_logprobs`)、`src/post_training/inference.py`(`generate_reply` / `load_model_from_ckpt`)、`scripts/chat.py`(命令行入口)、`src/models/transformer.py`(最朴素的 `generate`)
- 前置回顾:[第 06 章 · 因果掩码](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-06-attention)、[第 07 章 · softmax 与标签错位](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-07-objectives)

下一章起进入"预训练实战"阶段:先搞定数据——把 The Pile 语料处理成模型能高效读取的扁平 token。

下一章 👉 [第 10 章:数据流水线 · 从 The Pile 到 HDF5](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-10-data-pipeline)

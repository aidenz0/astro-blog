---
title: "从零训练大模型（八）：优化与训练系统"
author: Aidenz
pubDatetime: 2026-07-13T08:10:00Z
slug: train-llm-scratch-08-optimization
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 8
tags:
  - LLM
  - 大模型
  - 从零训练
  - 优化器
description: "训练系统全景：从梯度下降到 Adam、AdamW，学习率调度、梯度累积、混合精度与分布式数据并行（DDP），把“怎么把模型训得动、训得稳”讲透。"
---

> **本章前置**:第 02 章(导数与梯度、梯度下降)、第 03 章(PyTorch 训练循环五件套:前向 → 算损失 → 反向 → 更新 → 清零)、第 07 章(交叉熵损失)。
>
> **你将学到**:为什么"原始梯度下降"在真实大模型上很难用;**动量**怎么给梯度加上"惯性";**RMSProp** 怎么按方向自适应缩放步长;**Adam / AdamW** 的一阶矩、二阶矩、偏差校正怎么一步步推出来,以及 AdamW 为什么要把权重衰减"解耦";**学习率调度**(warmup + 余弦)为什么必不可少;以及四件工程"放大器"——**梯度累积、混合精度(bf16)、梯度裁剪、DDP 多卡**——如何把第 03 章那个玩具训练循环,扩展成能真正训练一个 ~400M 模型的训练系统。
>
> 👈 [上一章:训练目标 · 交叉熵与困惑度](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-07-objectives) ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

到第 07 章为止,我们已经有了一个完整的"目标":给定一批文本,模型前向算出 logits,交叉熵告诉我们"预测有多差",反向传播算出每个参数的梯度 $g$。

剩下的问题听起来很简单:**梯度有了,怎么更新参数?**

第 02 章给过最朴素的答案——梯度下降:

$$
\theta_{t+1} = \theta_t - \eta\, g_t
$$

逐符号:$\theta_t$ 是第 $t$ 步的参数,$g_t$ 是这一步算出的梯度,$\eta$(读作 "eta")是**学习率**(步长),减号表示"往梯度的反方向走"(梯度指向损失上升最快的方向,我们要下降)。

这个公式在第 03 章的玩具任务上工作得很好。但当你真的拿它去训一个有几亿参数、在海量噪声数据上跑几万步的 Transformer 时,它会处处碰壁。本章就是讲:工程界是怎么一步步把这行公式,补成今天大模型实际在用的那套优化器和训练系统的。

本仓库实际使用的"全家桶"(都在 `src/post_training/optim.py` 和 `scripts/pretrain_base.py` 里)是:

- **AdamW** 优化器;
- **线性 warmup + 余弦衰减**的学习率调度;
- **梯度累积**(用小 batch 模拟大 batch);
- **梯度裁剪**(防梯度爆炸);
- **bf16 混合精度**(更快更省显存);
- **DDP**(多卡数据并行)。

我们逐个拆开讲。

## 一、原始梯度下降的三个毛病

先把"为什么不能只用 $\theta_{t+1} = \theta_t - \eta g_t$"说清楚,后面每个改进才有动机。

**毛病 1:学习率极难调。** $\eta$ 太大,参数一步迈过最低点、来回震荡甚至发散(损失变成 `NaN`);$\eta$ 太小,训练慢得令人绝望。而"刚刚好"的 $\eta$,在训练早期和后期还不一样——早期想快,后期想稳。一个固定的 $\eta$ 没法同时满足。

**毛病 2:不同方向的"尺度"差别巨大。** 损失函数在不同参数方向上的陡峭程度天差地别。想象一个又长又窄的山谷:沿谷底方向很平缓,横跨山谷方向却很陡。用同一个 $\eta$,横向会剧烈来回弹跳,纵向却几乎不动。结果就是"在该慢的方向上太快,在该快的方向上太慢"。

**毛病 3:梯度噪声大。** 我们每步只用一个 mini-batch 估计梯度,而不是全量数据。这个估计是带噪声的,梯度方向时刻在抖。朴素更新会把这些抖动原封不动地传给参数,轨迹歪歪扭扭。

接下来的三个改进——动量、RMSProp、Adam——恰好分别对症下药:动量治"噪声和纵向太慢",RMSProp 治"各方向尺度不同",Adam 把两者合二为一。

## 二、动量:给梯度加上"惯性"

**直觉**:把参数想象成一个在损失曲面上往下滚的小球。朴素梯度下降的"球"没有质量——每一步只看当前脚下的坡度,坡度一抖它就跟着抖。**动量(momentum)**给小球加上质量:它会记住"之前一直在往哪个方向滚",用历史方向给当前方向"投票"。一致的方向被不断累加、越滚越快;来回乱抖的噪声方向则相互抵消。

数学上,动量维护一个**梯度的指数滑动平均**(exponential moving average,EMA)$m_t$:

$$
m_t = \beta\, m_{t-1} + (1-\beta)\, g_t
$$

$$
\theta_{t+1} = \theta_t - \eta\, m_t
$$

逐符号:$m_t$ 是"平滑后的梯度",$g_t$ 是当前这步的原始梯度,$\beta$(典型值 0.9)是"记忆系数"。

怎么理解这个 EMA?把递推式展开一层一层代进去:

$$
m_t = (1-\beta)\big(g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots\big)
$$

也就是说,$m_t$ 是过去所有梯度的**加权平均**,越近的梯度权重越大($\beta^0=1$),越老的权重按 $\beta$ 的幂次指数衰减($\beta^k$ 随 $k$ 变小)。$\beta=0.9$ 时,大致相当于对最近约 $1/(1-\beta)=10$ 步的梯度求平均。

- 一致方向(真信号)被反复累加 → 步子变大,纵向不再"几乎不动";
- 抖动方向(噪声)正负相消 → 平均掉了,横向不再剧烈弹跳。

一句话:**动量 = 对梯度做时间上的平滑**,既加速又抗噪。

## 三、RMSProp:按方向自适应缩放步长

动量解决了"噪声和纵向太慢",但毛病 2(各方向尺度不同)还在:我们仍然对所有参数用同一个 $\eta$。

**RMSProp 的直觉**:能不能让"长期梯度一直很大"的方向自动迈小步,"长期梯度一直很小"的方向自动迈大步?也就是给每个参数配一个**专属的、自适应的学习率**。

办法是再维护一个量——**梯度平方的指数滑动平均** $v_t$,它衡量"这个方向最近的梯度有多大(的平方)":

$$
v_t = \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta\, \frac{g_t}{\sqrt{v_t}+\epsilon}
$$

逐符号:$g_t^2$ 是梯度逐元素平方(每个参数各算各的),$v_t$ 是它的滑动平均,$\sqrt{v_t}$ 近似"这个方向梯度的典型幅度",$\epsilon$(很小,如 $10^{-8}$)只是防止除以 0。

关键在那个除法 $g_t / \sqrt{v_t}$:

- 某个方向长期梯度很大 → $\sqrt{v_t}$ 大 → 除完后步子被**压小**;
- 某个方向长期梯度很小 → $\sqrt{v_t}$ 小 → 除完后步子被**放大**。

效果是:不管原始梯度的尺度差多少,**除以各自的幅度后,所有方向的有效步长被拉回到差不多的量级**。那个又长又窄的山谷被"重新缩放"成了一个近似圆形的碗,沿任何方向下降都顺畅了。这就是"自适应学习率"。

## 四、Adam:把动量和 RMSProp 合二为一

到这里你大概已经猜到了:动量平滑了"往哪走",RMSProp 缩放了"走多大",**为什么不两个一起用?** 这正是 **Adam**(Adaptive Moment Estimation)做的事。

Adam 同时维护两个滑动平均——梯度的(一阶矩)和梯度平方的(二阶矩):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t \qquad\text{(一阶矩:平滑后的方向,来自动量)}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \qquad\text{(二阶矩:梯度幅度,来自 RMSProp)}
$$

"矩(moment)"是统计学术语:一阶矩约等于均值,二阶矩约等于(未中心化的)方差。所以 $m_t$ 估计"梯度的平均方向",$v_t$ 估计"梯度的平均能量"。

### 偏差校正:开头为什么要"放大"

这里有一个细节,新手最容易忽略却很重要。我们把 $m_0$ 和 $v_0$ 都初始化为 0。那么第 1 步:

$$
m_1 = \beta_1 \cdot 0 + (1-\beta_1) g_1 = (1-\beta_1) g_1
$$

$\beta_1=0.9$ 时,$m_1 = 0.1\, g_1$——只有真实梯度的十分之一!因为滑动平均刚启动,被那个 0 的初值"往下拽"了。训练最初几十步,$m_t$ 和 $v_t$ 都被系统性地**低估**,这叫"偏差(bias)"。

Adam 用一个干净的公式校正它。可以证明,在梯度大致稳定时,$m_t$ 的期望约为真实值的 $(1-\beta_1^t)$ 倍。所以只要除以这个因子就能"放大回去":

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}
$$

逐符号:$t$ 是步数(从 1 开始),$\beta_1^t$ 是 $\beta_1$ 的 $t$ 次方。第 1 步时分母 $=1-\beta_1=0.1$,把那个被压小 10 倍的 $m_1$ 正好放大 10 倍补回来;随着 $t$ 增大,$\beta_1^t \to 0$,分母 $\to 1$,校正自动消失——因为这时滑动平均已经"热身"完毕,不需要补了。

### 最终更新式

把校正后的一阶矩当"方向"、校正后的二阶矩当"缩放",合成 Adam 的更新:

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

逐符号读一遍:用平滑后的方向 $\hat m_t$(抗噪、有惯性),除以该方向的典型幅度 $\sqrt{\hat v_t}$(各方向尺度拉平),乘学习率 $\eta$,反方向更新。三个毛病一次性都照顾到了。

典型超参:$\beta_1=0.9$(方向记忆约 10 步)、$\beta_2=0.999$ 或本仓库的 **0.95**、$\epsilon=10^{-8}$。本仓库 `src/post_training/optim.py` 里就把 `betas` 设成了 `(0.9, 0.95)`:

```python
def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
```

($\beta_2$ 用 0.95 而不是默认 0.999,是大模型预训练常见的选择:对二阶矩的记忆短一点,对梯度幅度的突变更敏感、更稳。)

## 五、AdamW:把权重衰减"解耦"

我们想给损失加一个"别让权重长得太大"的正则项,这叫**权重衰减(weight decay)**,有助于泛化。老办法是 **L2 正则**:在损失里加一项 $\frac{\lambda}{2}\|\theta\|^2$。对它求导,会在梯度里多出一项 $\lambda\theta$:

$$
g_t \;\leftarrow\; g_t + \lambda\, \theta_t
$$

问题来了:这个 $\lambda\theta$ 也会一起被塞进 Adam 的 $m_t, v_t$,再被 $\sqrt{\hat v_t}$ **除一遍**。结果是——梯度本来就大的参数,它的衰减被这个除法削弱;梯度小的参数,衰减又被放大。**衰减力度变得和梯度幅度纠缠在一起,不再是我们想要的"对每个权重一视同仁地往 0 拉一点点"。**

**AdamW 的修正(W = Weight decay 解耦)**:别把衰减混进梯度,而是在参数更新这一步**单独、直接**地减一刀:

$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\, \theta_t\right)
$$

对比一下两者的差别,核心就一句:

- **Adam + L2**:$\lambda\theta$ 进梯度 → 被 $m_t,v_t$ 平滑、被 $\sqrt{\hat v_t}$ 缩放;
- **AdamW**:$\lambda\theta$ 不进梯度 → 绕开自适应缩放,直接、干净地把权重按比例往 0 拉。

这就是"解耦(decoupled)"的含义:让 Adam 的自适应部分只管"梯度方向上的优化",让权重衰减只管"正则化",互不污染。实践证明 AdamW 的泛化更好,已成为训练 Transformer 的事实标准,**本仓库正是用 AdamW**(`torch.optim.AdamW`)。

### 一个工程细节:不是所有参数都该衰减

打开 `src/post_training/optim.py`,你会看到它没有对全部参数无脑加衰减,而是分成两组:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() >= 2:
        decay.append(p)
    else:
        no_decay.append(p)
groups = [
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
]
```

判据是 `p.dim() >= 2`:维度 ≥ 2 的(各种权重**矩阵**,如线性层、注意力投影、嵌入矩阵)施加衰减;维度为 1 的(偏置 bias、LayerNorm 的缩放/平移系数等)**不衰减**。

为什么?像 LayerNorm 的缩放系数、偏置这种一维参数,本身就承担着"调整每个通道幅度/平移"的职责,把它们往 0 拉只会损害模型表达能力,带来的正则收益却几乎没有。这是 GPT 系列沿用至今的"标准配方"。注释里写得很直白:`Standard GPT recipe`。

## 六、学习率调度:warmup + 余弦衰减

毛病 1 说过:理想的学习率在训练早期和后期不一样。于是我们不再用固定 $\eta$,而是让它**随步数变化**,这叫**学习率调度(LR schedule)**。本仓库用的是经典的"**线性 warmup + 余弦衰减**",实现就在 `src/post_training/optim.py` 的 `cosine_lr` 里:

```python
def cosine_lr(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)
```

它分两段。

**第一段:线性 warmup(预热)。** 在前 $S_{\text{warmup}}$ 步里,把学习率从 0 线性升到峰值 $\eta_{\max}$:

$$
\eta(s) = \eta_{\max}\cdot\frac{s+1}{S_{\text{warmup}}}, \qquad s < S_{\text{warmup}}
$$

(代码里 `lr` 就是峰值 $\eta_{\max}$。)

**为什么需要 warmup?** 训练刚开始,权重是随机初始化的,梯度方向很不可靠,而 Adam 的二阶矩 $v_t$ 还没"热身"(回忆偏差校正:前几步 $\hat v_t$ 的估计很不稳)。如果一上来就用峰值学习率,很容易迈出一个灾难性的大步,把训练直接带崩(损失 `NaN`)。warmup 就像运动前的热身:先用很小的步子让 $m_t, v_t$ 和权重都进入"正常工作状态",再逐渐加速。

**第二段:余弦衰减(cosine decay)。** warmup 结束后,把学习率沿一条余弦曲线,从 $\eta_{\max}$ 平滑降到 $\eta_{\min}$:

$$
\eta(s) = \eta_{\min} + \tfrac{1}{2}\big(1+\cos(\pi p)\big)(\eta_{\max}-\eta_{\min}),
\quad p = \frac{s - S_{\text{warmup}}}{S_{\max} - S_{\text{warmup}}}
$$

逐符号:$p$ 是"进度",从 warmup 刚结束时的 0 走到训练末尾的 1。代入看两端:
- $p=0$:$\cos 0 = 1$,系数 $\tfrac12(1+1)=1$,得 $\eta_{\max}$(刚 warmup 完,满速);
- $p=1$:$\cos\pi = -1$,系数 $\tfrac12(1-1)=0$,得 $\eta_{\min}$(训练末尾,最慢)。

中间是一条先缓、中间快、末尾又缓的余弦曲线。这种"先大步探索、后小步精修"的安排,实践中收敛又快又稳。代码里 `step >= max_steps` 之后直接返回 `min_lr`,保证越界也安全。

在 `scripts/pretrain_base.py` 的主循环里,每一步都先查一次当前学习率,再写进优化器的每个参数组:

```python
lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=cfg.train_steps,
               lr=cfg.lr, min_lr=cfg.min_lr)
for g in optimizer.param_groups:
    g["lr"] = lr
```

预训练配置的默认值(`config/post_training_config.py`)是 `warmup_steps=2000`、`lr=3e-4`(峰值)、`min_lr=3e-5`(末值),正好是上面两段公式的参数。

## 七、梯度累积:用小 batch 模拟大 batch

大模型喜欢"大 batch":一次看更多样本,梯度估计的噪声更小(回忆毛病 3),训练更稳。但 batch 一大,前向/反向要同时存下的中间激活值就越多,**显存可能直接爆掉**。

**梯度累积(gradient accumulation)**是个巧妙的折中:把一个"大 batch"在时间上拆成若干个能塞进显存的"微批次(microbatch)",**逐个前向+反向,把梯度一次次累加起来,攒够了再更新一次参数**。因为梯度是可加的,N 个微批次的梯度之和,数学上等价于把它们拼成一个大 batch 一次算出来的梯度。

有效 batch 的大小是三者相乘:

$$
B_{\text{effective}} = B_{\text{micro}} \times N_{\text{accum}} \times N_{\text{gpus}}
$$

逐符号:$B_{\text{micro}}$ 是单卡单次微批次大小(配置里的 `batch_size`),$N_{\text{accum}}$ 是累积步数(`grad_accum`),$N_{\text{gpus}}$ 是 GPU 数(下一节 DDP)。例如 `batch_size=24`、`grad_accum=8`、2 卡,有效 batch $= 24\times 8\times 2 = 384$ 条序列/步。`scripts/pretrain_base.py` 启动时会把这行算式打印出来给你确认。

来看实现(`scripts/pretrain_base.py` 主循环):

```python
optimizer.zero_grad(set_to_none=True)
accum_loss = 0.0
for micro in range(cfg.grad_accum):
    xb, yb = next(batch_iter)
    # Only sync grads on the last micro-step (DDP optimization).
    sync = (micro == cfg.grad_accum - 1) or not ctx.enabled
    cm = model.no_sync() if (ctx.enabled and not sync) else _nullcm()
    with cm, amp_autocast(cfg.amp_dtype, ctx.device):
        _, loss = model(xb, yb)
        loss = loss / cfg.grad_accum
    loss.backward()
    accum_loss += loss.item()

torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()
```

注意三个关键点:

1. **`zero_grad` 在循环外**:整轮微批次开始前才清零一次,这样每个微批次的 `loss.backward()` 才能把梯度**累加**到同一个 `.grad` 上(PyTorch 默认就是累加)。
2. **`loss = loss / cfg.grad_accum`**:每个微批次的损失先除以累积步数。因为我们要的是 N 个微批次的**平均**梯度(等价于大 batch 的梯度),而不是 N 倍的和;不除的话梯度尺度会大 N 倍,等于偷偷把学习率放大了 N 倍。
3. **`optimizer.step()` 在循环外**:攒满 N 个微批次的梯度后,才裁剪、更新一次。

这正是把第 03 章"五件套"中的"前向→算损失→反向"重复了 N 次,再统一"更新→清零"。

## 八、混合精度 AMP / bf16:更快更省

默认情况下参数和计算用的是 32 位浮点(fp32)。**混合精度(Automatic Mixed Precision,AMP)**的想法是:在前向计算里,把矩阵乘法这类"耐受低精度"的运算改用更短的浮点(本仓库用 **bf16**,16 位),只在必要处保留 fp32。

好处有两个:

- **更省显存**:16 位只占一半字节,中间激活值的内存大致砍半,于是能塞下更大的模型或更大的 batch;
- **更快**:现代 GPU(如 H100)对低精度矩阵乘有专门的硬件加速,吞吐量(tokens/秒)显著提升。

**为什么是 bf16 而不是 fp16?** 两者都是 16 位,但分配不同。`bf16` 保留了和 fp32 **一样宽的 8 位指数**,只牺牲尾数(有效数字)精度。指数宽意味着能表示的数值**范围**和 fp32 一样大,几乎不会发生上溢/下溢——这在深度学习里比"多几位有效数字"重要得多。fp16 指数只有 5 位,动态范围窄,经常需要额外的 "loss scaling" 技巧来防溢出;bf16 在 H100 上**不需要 GradScaler**,用起来干净得多。

本仓库用一个 `amp_autocast` 上下文管理器把前向包起来(见上一节代码里的 `with ... amp_autocast(cfg.amp_dtype, ctx.device)`)。`cfg.amp_dtype` 默认 `"bf16"`,设成 `None` 就退回纯 fp32。注意:**模型参数本身仍存为 fp32**,只是 autocast 区域内的部分算子临时用 bf16 跑,兼顾稳定与速度。

> **一个会反复出现的坑(后训练阶段尤其重要)**:bf16 牺牲的是有效数字精度。在 PPO/GRPO/DPO 里,我们要把"新策略的 log-prob"和"旧/参考策略的 log-prob"**相减**得到一个很小的差值。两个相近的大数用 bf16 相减,低位早被舍掉,差值会被严重的舍入误差污染,直接毒化训练信号。所以这些算法里,凡是要相减的 log-prob 一律**强制转回 fp32** 再算。看 `src/post_training/rollout.py`,每次取 log-prob 都写成 `logits.float()`:
>
> ```python
> logprobs_all = F.log_softmax(logits.float() / max(temperature, 1e-6), dim=-1)
> ```
>
> 文件顶部注释也专门点明了这条规则:`log-probs are always taken in fp32 ... because PPO/GRPO/DPO subtract log-probs and bf16 rounding there is harmful`。这一点我们会在第 14、15、16 章再次遇到。

## 九、梯度裁剪:防止梯度爆炸

训练偶尔会撞上"坏数据"或"陡峭区域",某一步算出的梯度异常巨大。配上学习率一乘,就是一个灾难性的大步,可能把好不容易学到的权重一脚踹飞,损失瞬间变 `NaN`。

**梯度裁剪(gradient clipping)**给梯度的总长度设一个上限。它先算所有参数梯度拼起来的**全局范数(global norm)**$\|g\|_2$,如果超过阈值 $c$,就把整个梯度向量等比例缩小到长度恰好为 $c$:

$$
g \leftarrow g \cdot \min\!\left(1,\; \frac{c}{\|g\|_2}\right)
$$

逐符号:$\|g\|_2$ 是梯度的 L2 范数(整体长度),$c$ 是阈值。看那个 $\min$:

- 若 $\|g\|_2 \le c$:分式 $\ge 1$,取 1,梯度**原封不动**——正常情况什么都不做;
- 若 $\|g\|_2 > c$:取 $c/\|g\|_2 < 1$,梯度被按比例缩小,长度正好压到 $c$。

注意它**只改长度、不改方向**:整个向量同比缩放,各参数之间的相对比例不变,只是别让那一步迈得太狠。本仓库在 `optimizer.step()` 之前调用 `torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)`,默认 `grad_clip=1.0`。这是一道便宜的"安全护栏",在长序列和后续的 RL 训练里尤其救命。

## 十、DDP 多卡:数据并行

一张卡训得慢,我们想用多张卡一起训。最常用的方案是 **DistributedDataParallel(DDP)**,即**数据并行**。

它的工作方式很直观:

1. **每张 GPU 一个进程,各持一份完整的模型副本**(参数完全相同);
2. 每个进程喂入**不同的数据分片**(本仓库靠"每个 rank 用不同随机种子" `set_seed(cfg.seed + ctx.rank)` 让各卡看到不同数据流);
3. 各卡**独立**做前向、反向,各自算出一份梯度;
4. 更新参数前,把所有卡的梯度做一次 **all-reduce(求平均)**——每张卡都拿到"全体卡梯度的平均值";
5. 因为各卡梯度相同、初始参数也相同,各卡 `optimizer.step()` 后参数依然完全一致,无需额外同步。

第 4 步是关键:all-reduce 让 N 张卡的这一步**等效于把 N 份数据拼成一个大 batch**——这也是为什么有效 batch 公式里要乘 $N_{\text{gpus}}$。

**和梯度累积配合的小优化**:做梯度累积时,只有最后一个微批次才真正需要把梯度同步出去,前面几个微批次的同步是浪费带宽。所以本仓库在非最后的微步上用 `model.no_sync()` 关掉同步(回看第七节代码里的 `cm = model.no_sync() if (ctx.enabled and not sync) else _nullcm()`),只在最后一步触发 all-reduce。

**只让 rank0 干"独占"的事**:打印日志、跑评估记录、保存 checkpoint 这些只需做一次的事,统一交给主进程(rank0)。其余进程闷头算梯度就好,避免多份进程争抢着写同一个文件。`scripts/pretrain_base.py` 里随处可见 `if ctx.is_main:` 的守卫:

```python
if ctx.is_main and step > start_step and step % cfg.save_every == 0:
    save_stage_ckpt(cfg.out_ckpt, model, optimizer, stage="pretrain",
                    cfg=cfg, step=step, metrics={"train_loss": accum_loss})
```

启动命令也对应这两种模式(总览里约定过):

```bash
# 单卡
PYTHONPATH=. python scripts/pretrain_base.py
# 多卡(N=2):DDP + bf16,仅 rank0 记日志/存档
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py
```

## 小结:训练循环五件套的"工程扩展"

第 03 章的玩具训练循环是"五件套":**前向 → 算损失 → 反向 → 更新 → 清零**。本章做的,就是把每一件事都"加固"到能训真实大模型:

| 五件套环节 | 本章的扩展 |
|---|---|
| 前向 | 包在 **bf16 autocast** 里,更快更省;但 log-prob 相减处转回 fp32 |
| 算损失 | 梯度累积时 `loss / grad_accum`,凑出大 batch 的平均 |
| 反向 | 重复 N 个**微批次**累加梯度;DDP 在最后一个微步 **all-reduce** 求平均 |
| 更新 | 用 **AdamW**(动量 + 自适应缩放 + 解耦权重衰减),学习率走 **warmup+余弦**;更新前先**梯度裁剪** |
| 清零 | `zero_grad` 挪到整轮微批次之前,保证累加正确 |

把这五点连起来,你就读懂了 `scripts/pretrain_base.py` 主循环的每一行——下一章我们换个视角:模型训好了,怎么让它**生成**文本。

## 自测题

1. 朴素梯度下降的三个毛病分别是什么?动量、RMSProp、Adam 各主要治哪一个?
2. 写出 Adam 的 $m_t$ 和 $v_t$ 递推式。为什么需要偏差校正 $\hat m_t = m_t/(1-\beta_1^t)$?为什么随着 $t$ 增大这个校正会自动消失?
3. AdamW 和"Adam + L2 正则"在公式上到底差在哪一项?"解耦"解决了什么具体问题?
4. 本仓库为什么只对 `p.dim() >= 2` 的参数加权重衰减?哪些参数被排除了?
5. 为什么训练一开始要 warmup?如果跳过 warmup、直接用峰值学习率,最可能出什么事?
6. `batch_size=16`、`grad_accum=4`、4 张 GPU,有效 batch 是多少?为什么梯度累积时要把 `loss` 除以 `grad_accum`?
7. bf16 相比 fp16 好在哪?既然全程用了 bf16,为什么 PPO/GRPO/DPO 里的 log-prob 还要转回 fp32?
8. 梯度裁剪改变梯度的方向还是长度?阈值 `grad_clip=1.0` 时,范数为 0.5 和范数为 5 的梯度分别会发生什么?
9. DDP 里的 all-reduce 在做什么?为什么有了它,有效 batch 公式要乘上 GPU 数?为什么只让 rank0 存 checkpoint?

## 深入参考

- 工程速查:[`docs/zh/foundations/optimization_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/foundations/optimization_zh.md)(含训练监控指标表、显存调节杠杆顺序)
- 源码:`src/post_training/optim.py`(`configure_optimizer` / `cosine_lr`)、`scripts/pretrain_base.py`(完整训练主循环)、`config/post_training_config.py`(各阶段默认超参)
- 前置回顾:[第 02 章 · 梯度下降](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-02-math-ml-basics)、[第 03 章 · 训练循环](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-03-pytorch-intro)、[第 07 章 · 交叉熵](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-07-objectives)

下一章我们让训练好的模型开口说话:从 logits 到 token,贪心、温度、top-k、top-p 各有什么权衡。

下一章 👉 [第 09 章:生成与采样](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-09-generation)

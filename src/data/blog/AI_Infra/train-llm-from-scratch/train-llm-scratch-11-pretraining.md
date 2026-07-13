---
title: "从零训练大模型（十一）：预训练你的基座模型"
author: Aidenz
pubDatetime: 2026-07-13T08:07:00Z
slug: train-llm-scratch-11-pretraining
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 11
tags:
  - LLM
  - 大模型
  - 从零训练
  - 预训练
description: "动手跑通 pretrain_base.py：亲手预训练一个基座（Base）模型，并学会读懂 loss 与困惑度曲线，判断训练是否健康。"
---

> **本章前置**:第 03 章(最小训练循环五件套)、第 07 章(交叉熵与困惑度)、第 08 章(AdamW、学习率调度、梯度累积、DDP)、第 10 章(数据流水线)。
>
> **你将学到**:
> - 把前面所有零件拼成一个**完整的预训练循环**,并对照 `pretrain_base.py` 的真实代码逐段读懂;
> - 默认 ~400M 配置和 smoke 极小配置各自是干嘛的;
> - 怎么读训练日志:loss 和困惑度应该怎样下降,什么算正常、什么算异常;
> - **动手**:用 smoke 配置在小数据上跑几步,验证"训练能动起来";
> - **零成本动手**:一段**完全自包含、CPU 几分钟跑完、不依赖任何下载**的迷你预训练,让你亲眼看到 loss 下降、并用模型续写几个 token。
>
> 👈 [上一章:数据流水线](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-10-data-pipeline) ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

这是激动人心的一章。前面我们造了模型(第 05、06 章)、定义了损失(第 07 章)、备好了优化器(第 08 章)、炼好了数据(第 10 章)。现在,把它们**全部接在一起**,让模型第一次真正"开始学习"。

![预训练循环](./img/02_pretraining.png)

这张图就是本章的全部:**数据窗口 → 前向 → 交叉熵 → 反向 → AdamW(余弦 LR)→ 每隔若干步存 checkpoint**,然后回到第一步,循环几万次。我们先把这台机器在源码里看清楚,再亲手让它转起来。

## 11.1 完整预训练循环:逐段读懂 `pretrain_base.py`

主角是 [`scripts/pretrain_base.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/pretrain_base.py)。它本质上就是第 03 章那个"训练循环五件套"(取数据 → 前向算 loss → 反向 → 更新 → 清零)的"工业级加强版",多加了几样真正训练大模型时不可或缺的东西。我们顺着代码走。

### 第一步:读配置

```python
cfg, extras = parse_config_with_json(
    PretrainConfig, "configs/pretrain.json",
    extra={"--resume": dict(type=str, default=None, help="checkpoint to resume from")})
```

`pretrain_base.py` 不把超参数写死在代码里,而是用 [`src/post_training/cli.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/cli.py) 把配置层层叠起来。优先级**从低到高**是:

1. dataclass 里的默认值(`config/post_training_config.py` 的 `PretrainConfig`);
2. `configs/base.json`(共享的模型/运行时字段);
3. `--config` 指向的那个阶段 JSON(默认 `configs/pretrain.json`);
4. 命令行 `--字段 值` 覆盖(最高优先级)。

这意味着**几乎每个超参数都能从命令行直接覆盖**,比如 `--batch_size 8 --train_steps 50000`。这一点对下面的动手环节很关键。

> **小技巧**:任何时候想知道"我这条命令最后到底用了什么配置",加上 `--print-config` 就会把解析后的完整配置打印出来再退出,什么都不会真的运行:
> ```bash
> PYTHONPATH=. python scripts/pretrain_base.py --config configs/smoke/pretrain.json --print-config
> ```

### 第二步:初始化分布式、建模型、建优化器

```python
ctx = ddp_setup(cfg.device)
set_seed(cfg.seed + ctx.rank)        # 每个 rank 不同种子 → 看到不同的数据窗口

model = build_model_from_config(cfg).to(ctx.device)
...
model = ddp_wrap(model, ctx)
optimizer = configure_optimizer(unwrap(model), cfg.lr, cfg.weight_decay)
```

- `ddp_setup` 处理多卡(单卡时它就是个空操作,所以同一份代码单卡多卡都能跑)。
- `set_seed(cfg.seed + ctx.rank)`:还记得第 10 章的伏笔吗?每个 GPU 加上自己的 `rank` 当种子,于是 `get_batch_iterator` 里的全局随机洗牌在每张卡上结果不同,**不同卡看到不同窗口**,等于变相扩大数据。
- `build_model_from_config` 就是用配置里的 `n_head / n_embed / context_length / vocab_size / n_blocks` 实例化一个第 05 章的 `Transformer`。
- `configure_optimizer`([`optim.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/optim.py))是第 08 章讲的**带权重衰减拆分的 AdamW**:只对二维权重矩阵施加 `weight_decay`,绝不对偏置 / LayerNorm / embedding 施加(标准 GPT 配方)。

### 第三步:主循环——五件套 + 四样加强件

```python
batch_iter = get_batch_iterator(cfg.train_path, cfg.batch_size, cfg.context_length, device=ctx.device)

for step in range(start_step, cfg.train_steps):
    # 【加强件 1】余弦学习率:先线性预热,再余弦衰减
    lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=cfg.train_steps,
                   lr=cfg.lr, min_lr=cfg.min_lr)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    # 【加强件 2】梯度累积:攒 grad_accum 个 micro-batch 再更新一次
    for micro in range(cfg.grad_accum):
        xb, yb = next(batch_iter)
        sync = (micro == cfg.grad_accum - 1) or not ctx.enabled
        cm = model.no_sync() if (ctx.enabled and not sync) else _nullcm()
        # 【加强件 3】bf16 自动混合精度
        with cm, amp_autocast(cfg.amp_dtype, ctx.device):
            _, loss = model(xb, yb)               # 前向 + 交叉熵(第 07 章)
            loss = loss / cfg.grad_accum
        loss.backward()                            # 反向
        accum_loss += loss.item()

    # 【加强件 4】梯度裁剪:把梯度范数限制在 grad_clip 内,防止偶发爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()                               # 更新参数
```

把它和第 03 章的五件套对一下,你会发现骨架**一模一样**:`next(batch_iter)` 取数据 → `model(xb, yb)` 前向算 loss → `loss.backward()` 反向 → `optimizer.step()` 更新 → `zero_grad` 清零。四样加强件只是让它在大规模下更稳更快:

| 加强件 | 干什么 | 在第几章讲过 |
|---|---|---|
| 余弦 LR + 预热 (`cosine_lr`) | 开头几步小步快走(预热),之后学习率沿余弦曲线慢慢降到 `min_lr` | 第 08 章 |
| 梯度累积 (`grad_accum`) | 攒 `grad_accum` 个小 batch 的梯度再更新一次,等效于用大 batch 但省显存 | 第 08 章 |
| bf16 混合精度 (`amp_autocast`) | 前向用 bf16 算得快、省显存;bf16 不需要 `GradScaler`,主权重仍是 fp32 | 第 08 章 |
| 梯度裁剪 (`clip_grad_norm_`) | 把梯度整体范数压到 `grad_clip`(默认 1.0)以内,防止某一步梯度突然爆炸毁掉训练 | 第 08 章 |

> **`loss / cfg.grad_accum` 为什么要除?** 因为我们把 `grad_accum` 个小 batch 的梯度加在了一起,相当于求了"和";除以 `grad_accum` 把它变回"平均",这样无论 `grad_accum` 设多少,梯度的量级都一致,学习率不用跟着改。

### 第四步:日志、评估、存档

```python
if ctx.is_main and step % 20 == 0:
    print(f"step {step} | loss {accum_loss:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")

if step > start_step and step % cfg.eval_steps == 0:
    ev = estimate_loss(model, cfg, ctx, cfg.eval_iters)   # 在 train/dev 上各考一次
    ...

if ctx.is_main and step > start_step and step % cfg.save_every == 0:
    save_stage_ckpt(cfg.out_ckpt, model, optimizer, stage="pretrain",
                    cfg=cfg, step=step, metrics={"train_loss": accum_loss})
```

- 每 20 步打印一次 `loss / lr / tok/s`;只有 rank 0(主进程)负责打印和存档,避免多卡重复刷屏。
- 每 `eval_steps` 步调一次 `estimate_loss`,在 train 和 dev 两个数据上各算平均 loss(`@torch.no_grad()`,不更新参数,纯考试)。
- 每 `save_every` 步把模型 + 优化器 + 配置一起存进 `cfg.out_ckpt`(默认 `/ephemeral/ckpts/base_pretrained.pt`)。**checkpoint 里带着 `cfg`**,所以后续每个后训练阶段都能用它原样重建出同一个模型(参见 `load_backbone_from_ckpt`)。

这就是全部了。一个加了四样加强件的五件套,循环几万步——这就是"预训练"。

## 11.2 默认 ~400M 配置 vs smoke 极小配置

同一份 `pretrain_base.py`,喂不同的配置 JSON,就能是"真训练"或"跑通验证"两种完全不同的用途。

### 默认 ~400M(真训练)

模型架构来自 `config/post_training_config.py` 的 `BaseModelConfig`,超参数来自 `configs/pretrain.json`:

| 字段 | 值 | 含义 |
|---|---|---|
| `n_embed` | 1024 | 嵌入维度 |
| `n_head` | 16 | 注意力头数 |
| `n_blocks` | 24 | Transformer 层数 |
| `context_length` | 1024 | 上下文窗口 |
| `vocab_size` | 50304 | 词表大小 |
| `batch_size` / `grad_accum` | 8 / 12 | 每卡微批 / 累积步数 |
| `train_steps` | 50000 | 总训练步数 |
| `lr` / `min_lr` / `warmup_steps` | 3e-4 / 3e-5 / 2000 | 学习率峰值 / 谷值 / 预热步数 |

这套配置约 **4.06 亿(~400M)参数**。它需要**大数据 + 多张高端 GPU(如 2×H100)+ 数天**才能训出一个有用的基座。上下文从原始的 512 提到 1024,是为了之后能塞下 GSM8K 的推理链。

### smoke 极小(跑通验证)

[`configs/smoke/pretrain.json`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/configs/smoke/pretrain.json) 只改训练超参(`train_steps: 20` 等),而**模型维度由它旁边的 `configs/smoke/base.json` 自动接管**(配置加载器会优先用同目录的 `base.json`):

```json
// configs/smoke/base.json
{ "vocab_size": 50304, "context_length": 256, "n_embed": 128, "n_head": 4, "n_blocks": 2, "device": "cpu", "amp_dtype": null }
```

```json
// configs/smoke/pretrain.json
{ "batch_size": 4, "grad_accum": 1, "train_steps": 20, "eval_steps": 10000,
  "warmup_steps": 2, "save_every": 10000,
  "train_path": "/ephemeral/data/pile_train.h5", "dev_path": "/ephemeral/data/pile_dev.h5" }
```

模型缩到 128 维 / 4 头 / 2 层 / 256 上下文、`device=cpu`、`amp_dtype=null`(CPU 上不开混合精度)。它的用途**不是训出好模型**,而是**在 CPU 上几秒钟跑完 20 步,验证"整条流水线接得通、训练能动起来、不报错"**。这就是 "smoke test(冒烟测试)"的含义——通上电看冒不冒烟,先确认不会一开机就炸。

> **注意**:smoke 配置里的 `train_path` 仍指向 `/ephemeral/data/pile_train.h5`。也就是说,要跑这个官方 smoke,你**仍需先有那个 HDF5 文件**(哪怕很小)。如果你完全不想下载任何数据,直接跳到下面的 **11.5 零成本迷你预训练**——那段代码连数据文件都不需要。

## 11.3 怎么读训练日志

跑起来后,屏幕会持续刷出这样的行:

```
step 0 | loss 11.0600 | lr 1.50e-04 | 0 tok/s
step 20 | loss 9.8400 | lr 3.00e-04 | 31,840 tok/s
step 40 | loss 8.6100 | lr 3.00e-04 | 32,110 tok/s
...
  [eval] step 1000 | train 6.1200 | dev 6.2500
```

逐项看懂:

- **`loss`(最重要)**:滑动的交叉熵。它的**起点是 `ln(vocab) ≈ ln(50304) ≈ 10.8`**——这个数不是巧合:训练之初模型啥都不会,对每个位置只能在约 5 万个词里"瞎蒙",均匀瞎蒙的交叉熵恰好就是 `ln(词表大小)`(第 07 章推过)。**正常情况下 loss 应当稳步下降**,作者在 2×H100 上观察到的轨迹是 `11.06 → 8.6 → 6.0 → …`。这是最关键的单一健康信号:**只要 loss 在持续往下走,训练就是健康的。**

- **困惑度(perplexity)**:`困惑度 = exp(loss)`(第 07 章)。它的直观含义是"模型在每个位置平均要在多少个词里纠结"。loss=10.8 → 困惑度 ≈ 49000(几乎在整个词表里瞎蒙);loss=6.0 → 困惑度 ≈ 403(已经能把候选缩小到几百个),loss=4.0 → 困惑度 ≈ 55。**看着困惑度从几万掉到几百,就是模型从"文盲"变"识字"的过程。**

- **`lr`**:当前学习率。开头 `warmup_steps` 步内从小往上线性爬(预热),到峰值后沿余弦曲线慢慢降。

- **`tok/s`**:吞吐量,每秒处理多少 token,衡量速度(2×H100 合计约 32k/s)。它不反映模型好坏,只反映快慢。

- **`eval train` / `eval dev`**:在留出窗口上的平均 loss。**重点盯 `dev`**:如果 `train` 一直降但 `dev` 开始回升,就是**过拟合**的信号(模型在死记训练数据而非学习通用规律)。

**什么算异常?**

| 现象 | 大概率原因 |
|---|---|
| loss 变成 `nan` 或 `inf` | 学习率太大 / 数值爆炸;先调小 `lr`,确认 `grad_clip` 生效 |
| loss 卡在 10.8 附近不动 | 数据没真正喂进去、学习率几乎为 0、或标签错位写反了 |
| loss 先降后突然飙升 | 某一步梯度爆炸;梯度裁剪 (`grad_clip`) 正是为此而设 |
| `train` 降、`dev` 升 | 过拟合(数据太少 / 训太久) |

## 11.4 动手 A:用 smoke 配置跑通官方流水线

如果你已经有一个(哪怕很小的)`pile_train.h5`(见第 10 章),就能用官方 smoke 配置在 CPU 上跑通真正的 `pretrain_base.py`:

```bash
# 先看一眼解析后的完整配置(不会真的训练)
PYTHONPATH=. python scripts/pretrain_base.py --config configs/smoke/pretrain.json --print-config

# 真正跑 20 步 smoke(CPU,几秒~一两分钟)
PYTHONPATH=. python scripts/pretrain_base.py --config configs/smoke/pretrain.json
```

> 上面的 flag **逐字核对过源码**:脚本通过 `parse_config_with_json` 接收 `--config`,并允许用 `--任意字段` 覆盖配置(如 `--train_steps 50`、`--batch_size 2`)。

**多卡真训练**(需要 2 张 GPU + 真实 Pile 数据,数天量级):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py \
  --config configs/pretrain.json --batch_size 8 --grad_accum 12 --train_steps 50000
```

`torchrun --standalone --nproc_per_node=2` 启动 2 个数据并行的 rank(DDP + bf16),把 `nproc_per_node` 改成你的 GPU 数即可。有效 batch = `batch_size × grad_accum × GPU 数`。

> **为什么 `batch_size 8`?** 本仓库的教学版注意力会为每个 block 实例化一个 `(B, n_head, T, T)` 张量,显存主要被序列长度项吃掉。上下文 1024 时,batch 8 能舒服地装进一块 80GB H100,再用 `grad_accum` 把有效 batch 找回来。

## 11.5 动手 B:零成本迷你预训练(CPU,几分钟,无需任何下载)

完整预训练要大数据 + 多 GPU + 数天,大多数读者跑不动。但"预训练"这件事的**核心机理**——让模型在数据上反复跑五件套、亲眼看 loss 下降、再用它续写——其实可以浓缩成一段**完全自包含**的代码,在你的笔记本上几分钟跑完。

下面这段示例**不依赖项目的任何数据脚本、不下载任何东西**:它直接用 `tiktoken` 把一小段文本编码成 token,做成第 10 章那样的"扁平 token + 右移标签"数据,用 smoke 维度的项目原版 `Transformer`,套上第 03 章的训练循环五件套来训练。我们故意只用一小段**重复的文本**当语料——这样模型很容易就能学会其中的规律,你能在几百步内清楚看到 loss 砸下来。

把它存成 `mini_pretrain.py` 放在**项目根目录**,然后 `PYTHONPATH=. python mini_pretrain.py` 运行(`tiktoken` 在项目依赖里已经有了):

```python
"""一个完全自包含的迷你预训练示例:CPU 几分钟,看 loss 下降并续写几个 token。
放在项目根目录,运行:  PYTHONPATH=. python mini_pretrain.py
"""
import torch
import tiktoken
from src.models.transformer import Transformer

torch.manual_seed(0)
device = "cpu"

# --- 1. 数据:把一小段重复文本编码成扁平 token(对应第 10 章的"token 长河")---
enc = tiktoken.get_encoding("r50k_base")
text = ("the cat sat on the mat. the dog sat on the log. "
        "the cat ran to the dog. the dog ran to the cat. ") * 200
data = torch.tensor(enc.encode_ordinary(text), dtype=torch.long, device=device)
print(f"语料共 {len(data)} 个 token")

context_length = 32
batch_size = 16

def get_batch():
    # 随机选 batch_size 个起点,各切出 context_length+1 个 token(第 10 章的做法)
    ix = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])  # 标签=输入右移一位
    return x.to(device), y.to(device)

# --- 2. 模型:smoke 维度的项目原版 Transformer(参数与 build_model_from_config 一致)---
model = Transformer(
    n_head=4, n_embed=128, context_length=context_length,
    vocab_size=50304, N_BLOCKS=2,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数:{n_params/1e6:.1f}M")

# --- 3. 优化器:AdamW(第 08 章)---
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# --- 4. 训练循环五件套(第 03 章):取数据→前向算loss→反向→更新→清零 ---
model.train()
for step in range(800):
    xb, yb = get_batch()
    _, loss = model(xb, yb)        # forward 返回 (logits, loss),内部就是交叉熵
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 100 == 0 or step == 799:
        ppl = torch.exp(loss).item()
        print(f"step {step:4d} | loss {loss.item():.4f} | 困惑度 {ppl:8.1f}")

# --- 5. 用训好的模型续写几个 token,亲眼看看它"学到"了什么 ---
model.eval()
prompt = "the cat sat on"
idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=20)       # 项目原版的自回归生成
print("\n续写结果:")
print(enc.decode(out[0].tolist()))
```

**你会看到类似这样的输出**(具体数字因随机性略有不同):

```
语料共 9600 个 token
模型参数:13.1M
step    0 | loss 11.0xxx | 困惑度  60000.x
step  100 | loss  3.xxxx | 困惑度    2x.x
step  200 | loss  1.xxxx | 困惑度     x.x
...
step  799 | loss  0.xxxx | 困惑度     1.x

续写结果:
the cat sat on the mat. the dog sat on the log. the cat ...
```

这短短几十行,把这门课前十一章浓缩成了一次完整体验,请逐点对照:

1. **loss 从 ≈10.8 开始**(`ln(50304)`,第 07 章的"瞎蒙基线"),稳步往下砸——和真实预训练日志一模一样的形状,只是快了几万倍。
2. **困惑度从几万掉到接近 1**:因为语料是高度重复的,模型很快就"背会"了规律,在每个位置几乎不再纠结。真实语料做不到这么低(那意味着过拟合),但在这个玩具语料上,它直观展示了"模型从文盲到识字"的全过程。
3. **续写能接上**:喂 `the cat sat on`,模型续出 `the mat. the dog sat on the log...`——它确实从数据里学到了那段重复文本的结构。这就是预训练的产物:**一个会"预测下一个 token"的模型**。
4. 用的全是**项目真实组件**:`Transformer`、它的 `forward(xb, yb)`(内部交叉熵)、它的 `generate`。你在玩具数据上跑通的这套机理,和 `pretrain_base.py` 在 Pile 上跑的**完全是同一回事**,区别只在数据规模、模型大小和那四样工程加强件。

> **想多玩玩?** 把 `text` 换成你自己的一段文字、调大 `N_BLOCKS` 或 `n_embed`、改 `lr` 看 loss 曲线变化、把训练步数加到 2000——每一个改动都会让你对"超参数如何影响训练"多一分肌肉记忆。这正是这门课希望你养成的习惯:**改一个数字,看结果怎么变。**

## 小结

- `pretrain_base.py` 的主循环 = 第 03 章的**五件套**(取数据→前向算 loss→反向→更新→清零)+ **四样加强件**(余弦 LR 预热、梯度累积、bf16 混合精度、梯度裁剪),循环几万步,定期评估和存 checkpoint。
- **配置分层**:dataclass 默认 < `base.json` < 阶段 JSON < 命令行 `--字段`;`--print-config` 可先验证再跑。
- **默认 ~400M**(1024/16/24/1024)要大数据 + 多 GPU + 数天;**smoke**(128/4/2/256/cpu,20 步)只为在 CPU 上验证"流水线接得通"。
- 读日志:**loss 从 `ln(vocab)≈10.8` 起步、稳步下降**是最重要的健康信号;**困惑度 = exp(loss)** 直观表示"平均在多少词里纠结";盯 `dev` loss 防过拟合;`nan`、卡死不动、突然飙升都是异常。
- **11.5 的迷你示例**用项目真实 `Transformer` + tiktoken 编码的小文本,在 CPU 几分钟内完整复现了"loss 下降 + 模型续写",和真实预训练是同一套机理。

## 自测题

1. 训练之初 loss 大约是多少?为什么是这个值?(提示:第 07 章 + `vocab_size`)
2. 预训练循环里的"四样加强件"分别是什么?各解决什么问题?
3. `loss = loss / cfg.grad_accum` 这一行如果删掉,会有什么后果?
4. smoke 配置为什么能在 CPU 上几秒跑完,而默认配置要数天?至少说出三个让它"变小变快"的字段。
5. 如果你看到 `train` loss 持续下降但 `dev` loss 开始上升,发生了什么?该怎么办?
6. 在 11.5 的迷你示例里,如果把标签 `y` 改成和 `x` 完全相同(不右移),模型还能学会续写吗?为什么?

> 参考答案要点:① ≈ `ln(50304) ≈ 10.8`,因为初始模型只能在约 5 万词里均匀瞎蒙,均匀分布的交叉熵等于 `ln(词表大小)`。② 余弦 LR 预热(稳定地从小到大再衰减学习率)、梯度累积(用小显存等效大 batch)、bf16 混合精度(更快更省显存)、梯度裁剪(防梯度爆炸)。③ 梯度会被累加成"和"而非"平均",有效梯度量级随 `grad_accum` 放大,等价于学习率被偷偷乘大,容易训不稳甚至爆炸。④ 模型更小(n_embed=128、n_head=4、n_blocks=2)、上下文更短(256)、步数少(20)、device=cpu、不开混合精度等。⑤ 过拟合;可减少训练步数、增大数据量、加正则或早停。⑥ 学不会;y=x 等于让模型"原样复制输入",它永远学不到"根据前文预测下一个 token"的能力,续写会失败。

## 深入参考

- 工程速查:[`../02_pretraining_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/02_pretraining_zh.md)(2×H100 预训练配方的精炼版)。
- 第一性原理:[`../foundations/optimization_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/foundations/optimization_zh.md)、[`../foundations/objectives_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/foundations/objectives_zh.md)。
- 真实源码:[`scripts/pretrain_base.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/scripts/pretrain_base.py)、[`src/post_training/optim.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/optim.py)、[`config/post_training_config.py`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/config/post_training_config.py)。
- 命令与配置:[`../howto/train_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/howto/train_zh.md)、[`../howto/configs_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/howto/configs_zh.md)。

基座练成,模型会"说话"了——但它还不会"听话"。下一阶段,我们教它遵循指令。👉 [第 12 章:SFT 指令微调 · 含掩码损失推导](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-12-sft)

---
title: "从零训练大模型（一）：环境搭建与第一次运行"
author: Aidenz
pubDatetime: 2026-07-13T12:17:00Z
slug: train-llm-scratch-01-setup
featured: false
draft: false
series: 从零训练大模型
seriesOrder: 1
tags:
  - LLM
  - 大模型
  - 从零训练
  - PyTorch
description: "从零装好 Python 环境、把 train-llm-from-scratch 项目跑起来，认识它的目录结构与 smoke 小配置，完成你的第一次运行。"
---

> **本章前置**:你会一点点编程概念(知道"命令行""变量""函数"大致是什么),会在自己电脑上打开一个终端(Terminal / 命令提示符)。**不需要**任何机器学习或 PyTorch 基础。
>
> **你将学到**:为什么要给项目单独建一个"干净的 Python 小屋"(虚拟环境);怎么装 Python、建虚拟环境、把本项目装进去;这个项目的目录长什么样、"想干嘛该看哪个文件夹";最后**亲手跑通第一段代码**——构造一个迷你 Transformer、喂给它一批假数据、打印输出形状;再跑一遍项目自带的冒烟测试,并学会看懂常见报错。
>
> 👈 这是第一章,没有上一章 ｜ [返回总览](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-overview)

---

欢迎来到第一课!这一章我们先不碰任何高深的数学和模型原理,只做一件最实在的事:**把环境搭好,让代码在你的电脑上真正跑起来**。很多人学大模型卡在第一步——装不上、跑不通、报一堆看不懂的错——然后就放弃了。本章就是要帮你迈过这道坎。读完之后,你手上会有一个能用的环境,以及"我真的让一个 Transformer 算出了东西"的成就感。

我们全程**只用 CPU**(也就是你电脑里的普通处理器,没有显卡也没关系),用项目自带的"小号"配置,几秒钟就能跑完。

## 1. 为什么需要一个独立的 Python 虚拟环境

先打个生活化的比方。

假设你家厨房只有一套调料架。今天你做川菜,买了一堆花椒辣椒;明天你想做甜点,又买了香草糖、可可粉。如果所有东西都堆在同一个架子上,迟早会乱:做甜点时手一抖抓到花椒,或者两瓶不同牌子的盐放在一起,你都不知道该用哪瓶。

Python 项目也是这样。每个项目都依赖一堆**第三方库**(别人写好的现成代码包,比如我们要用的 `torch`、`numpy`)。不同项目可能需要**同一个库的不同版本**:项目 A 要 `torch` 2.0,项目 B 要 `torch` 2.5。如果你把所有库都装到"系统全局"那一个架子上,版本就会打架,装了 B 需要的版本,A 就跑不了了。

**虚拟环境(virtual environment,简称 venv)** 就是给每个项目单独配一套调料架:

- 它是一个独立的文件夹,里面装着这个项目专用的 Python 和所有库;
- 你在哪个项目里工作,就"激活"哪个虚拟环境,用的就是那一套库;
- 项目之间互不干扰,删掉一个虚拟环境也不会影响系统其它部分。

一句话:**虚拟环境让"这个项目需要什么"和"别的项目需要什么"互不打架**。这是 Python 开发的标准做法,养成习惯,以后会少踩无数坑。

## 2. 安装 Python

本项目要求 **Python 3.9 或更高版本**(这一点写在项目的 `pyproject.toml` 里,后面会讲到这个文件)。先确认你电脑上有没有合适的 Python。

打开终端,输入:

```bash
python3 --version
```

如果它打印出类似 `Python 3.11.5`(只要 `3.9` 及以上)就说明已经装好了,可以跳到下一节。

> **小提示**:有的系统里命令是 `python`,有的是 `python3`;有的 `python` 指向很老的 Python 2。本章统一用 `python3`,如果你的系统 `python` 就是 3.9+,把命令里的 `python3` 换成 `python` 也一样。

如果没装,或版本太低:

- **macOS**:推荐用 [Homebrew](https://brew.sh/),装好后运行 `brew install python`;或者去 [python.org](https://www.python.org/downloads/) 下载安装包。
- **Windows**:去 [python.org](https://www.python.org/downloads/) 下载安装包,**安装时务必勾选 "Add Python to PATH"**(把 Python 加入环境变量),否则终端找不到它。
- **Linux**:大多数发行版自带;用包管理器装,例如 Ubuntu 上 `sudo apt install python3 python3-venv python3-pip`。

## 3. 把项目拿到本地

如果你还没有项目代码,先用 `git` 克隆下来(没有 `git` 的话,也可以在 GitHub 页面上点 "Download ZIP" 下载解压):

```bash
git clone https://github.com/FareedKhan-dev/train-llm-from-scratch.git
cd train-llm-from-scratch
```

`cd` 是 "change directory"(切换目录)的缩写。执行后你就**站在项目根目录**里了——后面所有命令都默认在这个目录下运行,这一点很重要。

## 4. 创建并激活虚拟环境

在项目根目录里,创建一个名为 `venv` 的虚拟环境:

```bash
python3 -m venv venv
```

`python3 -m venv venv` 的意思是:用 Python 自带的 `venv` 模块,在当前目录下建一个叫 `venv` 的虚拟环境文件夹。这一步只需做一次。

然后**激活**它(每次开新终端工作前都要激活一次):

```bash
# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

激活成功后,你的命令行提示符前面会多出一个 `(venv)` 标记,像这样:

```
(venv) you@computer train-llm-from-scratch %
```

看到这个 `(venv)` 就说明:**从现在起,你装的库都进这个小屋,跑的 `python` 也是小屋里的那个**。想退出虚拟环境时,运行 `deactivate` 即可。

## 5. 安装本项目

项目根目录里有个文件叫 `pyproject.toml`,它是这个项目的"说明书":写清了项目叫什么、需要哪个版本的 Python、依赖哪些库。我们不用手动一个个装库,只要让 `pip`(Python 的包安装器)去读这份说明书自动安装就行。

确保虚拟环境已激活,然后运行:

```bash
pip install -e ".[train]"
```

这条命令有两个关键点,值得拆开讲:

- `-e` 是 **editable(可编辑)安装**。普通安装会把代码"复制一份"装到别处;可编辑安装则是"原地"安装——它让 Python **永远能找到你项目根目录里的代码**。这带来一个巨大的好处:**你不用再设置 `PYTHONPATH`**。在很多老教程里你会看到 `export PYTHONPATH=.` 这种操作,就是为了让 Python 找得到 `src/` 里的代码;有了可编辑安装,这一步就省了。而且你改了源码,改动会立刻生效,不必重装。
- `".[train]"` 里的 `[train]` 是**可选依赖组**。`pyproject.toml` 里把依赖分了组:核心依赖(`torch`、`numpy`、`h5py`、`tqdm`、`tiktoken` 等)总是会装;`[train]` 这一组额外装了后训练阶段要用的 `datasets`(下载 Alpaca、GSM8K 等公开数据集)和 `wandb`(可选的实验记录工具)。装上它,后面几章动手时就不缺东西了。

> **关于 PyTorch 的版本**:`pyproject.toml` 里只写了依赖 `torch`,**没有钉死 CUDA 版本**。如果你有 NVIDIA 显卡、想用 GPU,建议先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的指引装好对应 CUDA 的 `torch`,再运行上面的安装命令。**如果你只想用 CPU 跑本课的小例子,直接装就行**,默认会装一个能在 CPU 上工作的 `torch`。本课所有"动手"环节都设计成 CPU 也能跑。
>
> 另外项目里还有 `requirements.txt`(给原始预训练路径用,钉了 cu118 的 CUDA 轮子)和 `requirements-post.txt`(给后训练套件用,面向 CUDA 12.x)。这两个是给特定 GPU 环境的可选方案。**零基础、纯 CPU 学习,只用 `pip install -e ".[train]"` 这一条就够了。**

安装会下载一些库,稍等片刻。看到类似 `Successfully installed ...` 就成功了。

## 6. 项目目录导览

装好之后,我们花几分钟认识一下这个项目的"地形图"。理解每个文件夹的职责,你以后找东西、读代码会快很多。在项目根目录运行 `ls`(Windows 上用 `dir`)就能看到这些顶层文件夹。下面是**和本课最相关**的几个:

| 目录 / 文件 | 它负责什么 |
|---|---|
| `src/models/` | **模型本体**。Transformer、注意力、MLP、Transformer block 的定义都在这。核心文件 `src/models/transformer.py`。 |
| `src/post_training/` | **后训练 / 对齐**的全部算法:SFT、奖励模型、PPO、DPO、GRPO,以及推理、评估、生成等工具。 |
| `scripts/` | **各阶段的入口脚本**。每个训练阶段都有一个,例如 `scripts/pretrain_base.py`(预训练)、`scripts/train_sft.py`(SFT)、`scripts/train_dpo.py` 等;还有准备数据、对话、评估的脚本。 |
| `configs/` | **配置文件**(JSON 格式)。`base.json` 是模型基础尺寸,每个阶段一个文件(`sft.json`、`ppo.json`…);`configs/smoke/` 里是一套"小号"配置,专为在 CPU 上快速试跑而设。 |
| `config/` | 读取配置的代码(注意:`config/` 是 Python 代码,`configs/` 是 JSON 数据,别搞混)。`config/loader.py` 负责把 `configs/*.json` 加载进来。 |
| `data_loader/` | **数据加载**。把磁盘上的数据切成一批批喂给模型的工具,例如 `data_loader/data_loader.py`、`sft_dataset.py`、`preference_dataset.py`。 |
| `docs/` | **文档**。包括你正在读的这门中文教程(`docs/zh/tutorial/`)、工程速查参考(`docs/zh/`、`docs/zh/foundations/`)和英文 MkDocs 站点。 |
| `tests/` | **测试**。验证关键逻辑是否正确的小脚本,本章末尾我们会跑其中一个。 |
| `pyproject.toml` | 项目说明书(刚才用它装了项目)。 |

把这张表当成"想干嘛 → 看哪里"的速查:

| 你想干嘛 | 去看哪个目录 |
|---|---|
| 搞懂模型结构、改网络 | `src/models/` |
| 了解 SFT / PPO / DPO / GRPO 怎么实现 | `src/post_training/` |
| 真正启动某个阶段的训练 | `scripts/` |
| 调模型大小、学习率等参数 | `configs/`(数据)+ `config/`(读取代码) |
| 看数据是怎么被切批、喂进去的 | `data_loader/` |
| 读原理讲解和动手指引 | `docs/`(尤其本教程 `docs/zh/tutorial/`) |

后面的章节会带你深入其中每一块。现在,我们来跑第一段代码。

## 7. 第一次运行:让一个迷你 Transformer 算一次

是时候动手了!我们的目标很朴素:**按"小号"配置造一个 Transformer,喂给它一批随机的 token id,做一次前向计算,打印输出的形状**。你现在还不需要懂里面每个名词,本章只要"跑通、看到结果"。原理我们从第 02 章起一点点展开。

### 先看一眼配置

我们用的是项目自带的最小配置 `configs/smoke/base.json`。它的内容是:

```json
{
  "vocab_size": 50304,
  "context_length": 256,
  "n_embed": 128,
  "n_head": 4,
  "n_blocks": 2,
  "device": "cpu",
  "amp_dtype": null
}
```

逐个看这几个数字(现在只需有个模糊印象,后面章节会逐一深讲):

- `vocab_size`(词表大小)`50304`:模型认识多少种不同的 token。本项目用 OpenAI 的 `tiktoken` 分词器(`r50k_base` 编码),词表约 50304,特殊 token `<|endoftext|>` 的 id 是 50256。第 04 章细讲。
- `context_length`(上下文长度)`256`:模型一次最多能看多长的序列。
- `n_embed`(嵌入维度)`128`:每个 token 被表示成一个多长的向量。
- `n_head`(注意力头数)`4`:注意力机制里"并行视角"的数量。第 06 章细讲。
- `n_blocks`(层数)`2`:Transformer block 堆叠几层。
- `device`(设备)`"cpu"`:在 CPU 上跑,正合我们意。

对照一下:默认的"正经"约 400M 模型用的是 `n_embed=1024, n_head=16, n_blocks=24, context_length=1024`——比这个 smoke 配置大得多,需要 GPU 才跑得动。我们这里用的是它的"迷你版"。

### 写下并运行第一段脚本

在项目根目录新建一个文件 `first_run.py`(名字随意),内容如下。**这段代码会真正运行**——它用的构造参数和调用方式,完全来自 `src/models/transformer.py` 里 `Transformer` 类的真实定义。

```python
import json
import torch
from src.models.transformer import Transformer

# 1) 读取 smoke 小配置
with open("configs/smoke/base.json") as f:
    cfg = json.load(f)

# 2) 按配置构造一个迷你 Transformer
#    构造参数名严格对应 src/models/transformer.py 里 Transformer.__init__ 的签名:
#    n_head, n_embed, context_length, vocab_size, N_BLOCKS
model = Transformer(
    n_head=cfg["n_head"],
    n_embed=cfg["n_embed"],
    context_length=cfg["context_length"],
    vocab_size=cfg["vocab_size"],
    N_BLOCKS=cfg["n_blocks"],
)
model.eval()  # 推理模式(本例不训练,只做一次前向)

# 3) 造一批"假"的输入:2 个序列,每个长 16,里面是随机 token id
B, T = 2, 16
idx = torch.randint(0, cfg["vocab_size"], (B, T))
print("输入形状 (B, T):", tuple(idx.shape))

# 4) 前向计算。forward(idx, targets=None) 返回 (logits, loss);
#    不给 targets 时 loss 为 None。用 no_grad 省内存、不追踪梯度。
with torch.no_grad():
    logits, loss = model(idx)

# 5) 打印结果
print("输出 logits 形状 (B, T, vocab_size):", tuple(logits.shape))
print("loss:", loss)  # 没给 targets,所以是 None

# 顺手数一下这个迷你模型有多少参数
total = sum(p.numel() for p in model.parameters())
print("参数量:", f"{total:,}")
```

在**已激活虚拟环境**的终端里运行:

```bash
python3 first_run.py
```

你会看到类似这样的输出(参数量、loss 的具体值不重要,关键是形状):

```
输入形状 (B, T): (2, 16)
输出 logits 形状 (B, T, vocab_size): (2, 16, 50304)
loss: None
参数量: ...
```

恭喜——**你刚刚让一个 Transformer 完成了一次真实的前向计算!**

来读懂这个结果。我们喂进去的是形状 `(2, 16)` 的整数张量:2 个序列、每个 16 个 token。模型吐出的 `logits` 形状是 `(2, 16, 50304)`:对**每个序列的每个位置**,模型都给出了一组长度为 `50304` 的分数——也就是"接下来这个位置,词表里每个 token 各有多大可能"。这个 `(B, T, C)` 的形状(批大小、序列长度、通道/特征数)会贯穿整门课,务必记住它。我们还没训练,所以这些分数现在是"瞎猜"的,但通路已经打通了。

> **关于 `(B, T, C)`**:`B`=batch(一批里有几个序列),`T`=time / tokens(序列有多长),`C`=channels / 这里就是词表大小或嵌入维度。第 02 章会专门带你熟悉张量和这种形状记法,第 04、05 章会反复用到。

## 8. 跑一跑项目自带的冒烟测试

"冒烟测试(smoke test)"这个词来自硬件:新电路板通电,先看看会不会冒烟——如果连电都通不过,别的就不用谈了。软件里的冒烟测试同理:**用极小的输入、几秒钟,快速验证"核心逻辑没坏"**,不追求覆盖所有情况。

本项目在 `tests/test_post_training_smoke.py` 里有一组这样的测试。它**不下载数据、不真正训练**,只在 CPU 上用迷你模型验证一批"承重"逻辑是否正确——比如前向计算的两条路径是否给出一致结果、log 概率的计算和手算是否吻合、奖励解析是否正确等等。它是你检查"环境是否健康"的好工具。

运行它:

```bash
PYTHONPATH=. python3 tests/test_post_training_smoke.py
```

> **为什么这里又出现了 `PYTHONPATH=.`?** 因为这个测试脚本是被"直接当普通脚本"运行的,它需要从项目根目录找到 `src`、`config` 等包。`PYTHONPATH=.` 临时告诉 Python"也到当前目录找包"。其实你做了第 5 节的可编辑安装后,直接 `python3 tests/test_post_training_smoke.py` 通常也能跑通;加上 `PYTHONPATH=.` 只是更保险、对照官方写法。

如果一切正常,你会看到一连串以 `ok` 开头的行,最后是:

```
ok  forward_hidden matches forward
ok  compute_logprobs matches manual computation
...
ALL SMOKE TESTS PASSED
```

看到 `ALL SMOKE TESTS PASSED`,说明你的环境装对了、项目能正常工作。这是一个很好的"绿灯"信号,后面动手时如果遇到怪问题,可以回来先跑一遍它,确认底子没坏。

## 9. 常见报错排查

第一次搭环境难免磕磕绊绊。下面是几类最常见的报错和对策。

**(1) `ModuleNotFoundError: No module named 'src'` / `'torch'` / `'config'`**

"找不到某个模块"。分两种情况:

- 找不到 `torch`、`numpy` 这类**第三方库**:多半是**虚拟环境没激活**,或**没装项目**。检查命令行前面有没有 `(venv)`,没有就先 `source venv/bin/activate`;然后确认跑过 `pip install -e ".[train]"`。
- 找不到 `src`、`config` 这类**项目自己的包**:多半是**没在项目根目录运行**,或没做可编辑安装。先 `cd` 到项目根目录;确认做过可编辑安装。直接跑脚本时,可在命令前加 `PYTHONPATH=.`,例如 `PYTHONPATH=. python3 tests/test_post_training_smoke.py`。

**(2) `python: command not found` 或版本不对**

系统里命令可能是 `python3` 而不是 `python`,或 `python` 指向了太老的版本。统一用 `python3`,并用 `python3 --version` 确认是 3.9+。

**(3) 和 CUDA / GPU 有关的报错(如 `CUDA error`、`Torch not compiled with CUDA enabled`)**

这通常发生在配置里写了 `device: "cuda"` 但你的机器没有可用 NVIDIA 显卡,或者装的 `torch` 是 CPU 版。**本课所有小例子都不需要 GPU**:smoke 配置里 `device` 已经是 `"cpu"`,上面的 `first_run.py` 也没用到 GPU。只要确保用的是 CPU 路径,就能在普通笔记本上跑。真正的大规模训练才需要 GPU,届时我们会明确标注。

**(4) `pip install` 很慢或下载失败**

多半是网络问题(尤其下载 `torch` 这种大包时)。可以换用国内镜像源,例如:

```bash
pip install -e ".[train]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**(5) 权限相关报错(`Permission denied`)**

通常是因为没在虚拟环境里、试图往系统目录装东西。再次确认 `(venv)` 已激活——在虚拟环境里安装一般不需要管理员权限。

排查的总原则:**仔细读最后几行报错**,Python 的错误信息通常会直接点明"缺什么 / 在哪一行 / 是什么类型的错"。把关键词复制去搜索,往往很快有答案。

## 小结

- **虚拟环境**给每个项目一套独立的库,避免版本打架;用 `python3 -m venv venv` 创建、`source venv/bin/activate` 激活。
- 用 **`pip install -e ".[train]"`** 一条命令安装本项目:`-e` 是可编辑安装(省去 `PYTHONPATH`、改源码即时生效),`[train]` 装上后训练要用的数据集等依赖。
- 项目地形:`src/models/` 模型、`src/post_training/` 后训练、`scripts/` 各阶段入口、`configs/` 配置、`data_loader/` 数据、`docs/` 文档、`tests/` 测试。
- 你**亲手跑通了第一段代码**:按 `configs/smoke/base.json` 造迷你 Transformer,喂随机 token,前向得到 `(B, T, vocab_size)` 形状的 logits;并跑通了自带冒烟测试 `tests/test_post_training_smoke.py`。
- 常见报错里,"找不到模块"多半是环境没激活/没装/没在根目录;CUDA 报错用 CPU 路径即可绕开。

## 自测题

1. **为什么要为这个项目单独建一个虚拟环境?如果不建,可能出什么问题?**
   <details><summary>提示 / 答案</summary>不同项目可能依赖同一个库的不同版本。不隔离的话,装了 B 项目需要的版本会覆盖 A 需要的,导致 A 跑不了。虚拟环境给每个项目独立的一套库,互不干扰。</details>

2. **`pip install -e ".[train]"` 里的 `-e` 和 `[train]` 分别是什么意思?`-e` 帮我们省掉了什么麻烦?**
   <details><summary>提示 / 答案</summary>`-e` 是可编辑(editable)安装:Python 永远能找到项目根目录的代码,从而**省掉手动设置 `PYTHONPATH`**,且改源码即时生效。`[train]` 是可选依赖组,额外装上后训练用的 `datasets`、`wandb`。</details>

3. **第一段脚本里,模型输出 `logits` 的形状是 `(2, 16, 50304)`。这三个数字分别代表什么?**
   <details><summary>提示 / 答案</summary>`2` 是 batch 大小 `B`(一批 2 个序列);`16` 是序列长度 `T`(每个序列 16 个 token);`50304` 是词表大小,即对每个位置给出"词表里每个 token 各有多大可能"的一组分数。这就是贯穿全课的 `(B, T, C)` 形状。</details>

4. **跑冒烟测试时报 `ModuleNotFoundError: No module named 'src'`,你会先检查哪几件事?**
   <details><summary>提示 / 答案</summary>(a) 是否在项目根目录运行;(b) 虚拟环境是否激活(有没有 `(venv)`);(c) 是否做过 `pip install -e ".[train]"`;(d) 直接跑脚本时是否需要在命令前加 `PYTHONPATH=.`。</details>

5. **没有显卡,我能学完这门课、跑通本章的例子吗?**
   <details><summary>提示 / 答案</summary>能。本章及全课的动手小例子都设计成 CPU 可跑(smoke 配置 `device` 就是 `"cpu"`)。只有真正大规模预训练/对齐才需要 GPU,那些环节会被明确标注。</details>

## 深入参考

- 训练流程总览(UI 与 CLI、安装、输出去向):[`../howto/train_zh.md`](https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/docs/zh/howto/train_zh.md)
- 想用浏览器看带公式排版的文档站:`pip install -e ".[docs]"` 后运行 `mkdocs serve`。
- 模型源码,随时可对照:`src/models/transformer.py`。

---

理解了"环境怎么搭、代码怎么跑",下一章我们补上最关键的地基——一点点数学和机器学习直觉,让你真正看懂"模型在算什么、训练到底是怎么回事"。

下一章 👉 [第 02 章:数学与机器学习最小基础](/posts/ai_infra/train-llm-from-scratch/train-llm-scratch-02-math-ml-basics)

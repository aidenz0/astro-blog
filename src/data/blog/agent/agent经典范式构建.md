---
title: agent经典范式
author: Aidenz
pubDatetime: 2026-01-03T04:06:31Z
slug:
featured: false
draft: false
tags:
  - LLM
  - Agent
description: Agent经典范式构建
---

## 1. ReAct

最经典的一个智能体范式**ReAct (Reason + Act)**。ReAct由Shunyu Yao于2022年提出，其核心思想是模仿人类解决问题的方式，将**推理 (Reasoning)** 与**行动 (Acting)** 显式地结合起来，形成一个“思考-行动-观察”的循环。

### ReAct工作流程

流的方法可以分为两类：一类是“纯思考”型，如**思维链 (Chain-of-Thought)**，它能引导模型进行复杂的逻辑推理，但无法与外部世界交互，容易产生事实幻觉；另一类是“纯行动”型，模型直接输出要执行的动作，但缺乏规划和纠错能力。

  

ReAct的巧妙之处在于，它认识到**思考与行动是相辅相成的**。思考指导行动，而行动的结果又反过来修正思考。为此，ReAct范式通过一种特殊的提示工程来引导模型，使其每一步的输出都遵循一个固定的轨迹：

  

- **Thought (思考)：** 这是智能体的“内心独白”。它会分析当前情况、分解任务、制定下一步计划，或者反思上一步的结果。

- **Action (行动)：** 这是智能体决定采取的具体动作，通常是调用一个外部工具，例如 `Search['华为最新款手机']`。

- **Observation (观察)：** 这是执行`Action`后从外部工具返回的结果，例如搜索结果的摘要或API的返回值。

  

智能体将不断重复这个 **Thought -> Action -> Observation** 的循环，将新的观察结果追加到历史记录中，形成一个不断增长的上下文，直到它在`Thought`中认为已经找到了最终答案，然后输出结果。这个过程形成了一个强大的协同效应：**推理使得行动更具目的性，而行动则为推理提供了事实依据。**

  

![image.png](https://raw.githubusercontent.com/aidenz0/blogImage/main/img/20260119141053708.png)

这种机制特别适用于以下场景：

  

- **需要外部知识的任务**：如查询实时信息（天气、新闻、股价）、搜索专业领域的知识等。

- **需要精确计算的任务**：将数学问题交给计算器工具，避免LLM的计算错误。

- **需要与API交互的任务**：如操作数据库、调用某个服务的API来完成特定功能。

  

### ReAct智能体的实现

1. 系统提示词设计

提示词是整个 ReAct 机制的基石，它为大语言模型提供了行动的操作指令。我们需要精心设计一个模板，它将动态地插入可用工具、用户问题以及中间步骤的交互历史。

  

```

# ReAct 提示词模板

REACT_PROMPT_TEMPLATE = """

请注意，你是一个有能力调用外部工具的智能助手。

  

可用工具如下:

{tools}

  

请严格按照以下格式进行回应:

  

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。

Action: 你决定采取的行动，必须是以下格式之一:

- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。

- `Finish[最终答案]`:当你认为已经获得最终答案时。

- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

  

现在，请开始解决以下问题:

Question: {question}

History: {history}

"""

```

这个模板定义了智能体与LLM之间交互的规范：

  

- **角色定义**： “你是一个有能力调用外部工具的智能助手”，设定了LLM的角色。

- **工具清单 (`{tools}`)**： 告知LLM它有哪些可用的“手脚”。

- **格式规约 (`Thought`/`Action`)**： 这是最重要的部分，它强制LLM的输出具有结构性，使我们能通过代码精确解析其意图。

- **动态上下文 (`{question}`/`{history}`)**： 将用户的原始问题和不断累积的交互历史注入，让LLM基于完整的上下文进行决策。

2. 核心循环的实现

```python

class ReActAgent:

def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):

self.llm_client = llm_client

self.tool_executor = tool_executor

self.max_steps = max_steps

self.history = []

  

def run(self, question: str):

"""

运行ReAct智能体来回答一个问题。

"""

self.history = [] # 每次运行时重置历史记录

current_step = 0

  

while current_step < self.max_steps:

current_step += 1

print(f"--- 第 {current_step} 步 ---")

  

# 1. 格式化提示词

tools_desc = self.tool_executor.getAvailableTools()

history_str = "\n".join(self.history)

prompt = REACT_PROMPT_TEMPLATE.format(

tools=tools_desc,

question=question,

history=history_str

)

  

# 2. 调用LLM进行思考

messages = [{"role": "user", "content": prompt}]

response_text = self.llm_client.think(messages=messages)

if not response_text:

print("错误:LLM未能返回有效响应。")

break

  

# ... (后续的解析、执行、整合步骤)

```

3. 数据解析器的实现

```python

# (这些方法是 ReActAgent 类的一部分)

def _parse_output(self, text: str):

"""解析LLM的输出，提取Thought和Action。"""

thought_match = re.search(r"Thought: (.*)", text)

action_match = re.search(r"Action: (.*)", text)

thought = thought_match.group(1).strip() if thought_match else None

action = action_match.group(1).strip() if action_match else None

return thought, action

  

def _parse_action(self, action_text: str):

"""解析Action字符串，提取工具名称和输入。"""

match = re.match(r"(\w+)\[(.*)\]", action_text)

if match:

return match.group(1), match.group(2)

return None, None

```

  

- `_parse_output`： 负责从LLM的完整响应中分离出`Thought`和`Action`两个主要部分。

- `_parse_action`： 负责进一步解析`Action`字符串，例如从 `Search[华为最新手机]` 中提取出工具名 `Search` 和工具输入 `华为最新手机`。

4. 工具的调用和执行

```python

# (这段逻辑在 run 方法的 while 循环内)

# 3. 解析LLM的输出

thought, action = self._parse_output(response_text)

if thought:

print(f"思考: {thought}")

  

if not action:

print("警告:未能解析出有效的Action，流程终止。")

break

  

# 4. 执行Action

if action.startswith("Finish"):

# 如果是Finish指令，提取最终答案并结束

final_answer = re.match(r"Finish\[(.*)\]", action).group(1)

print(f"🎉 最终答案: {final_answer}")

return final_answer

tool_name, tool_input = self._parse_action(action)

if not tool_name or not tool_input:

# ... 处理无效Action格式 ...

continue

  

print(f"🎬 行动: {tool_name}[{tool_input}]")

tool_function = self.tool_executor.getTool(tool_name)

if not tool_function:

observation = f"错误:未找到名为 '{tool_name}' 的工具。"

else:

observation = tool_function(tool_input) # 调用真实工具

```

这段代码是`Action`的执行中心。它首先检查是否为`Finish`指令，如果是，则流程结束。否则，它会通过`tool_executor`获取对应的工具函数并执行，得到`observation`。

  

5. 观测结果的整合

最后一步，也是形成闭环的关键，是将`Action`本身和工具执行后的`Observation`添加回历史记录中，为下一轮循环提供新的上下文。

```python

# (这段逻辑紧随工具调用之后，在 while 循环的末尾)

print(f"👀 观察: {observation}")

# 将本轮的Action和Observation添加到历史记录中

self.history.append(f"Action: {action}")

self.history.append(f"Observation: {observation}")

  

# 循环结束

print("已达到最大步数，流程终止。")

return None

```

  

### ReAct

（1）ReAct 的主要特点

  

1. **高可解释性**：ReAct 最大的优点之一就是透明。通过 `Thought` 链，我们可以清晰地看到智能体每一步的“心路历程”——它为什么会选择这个工具，下一步又打算做什么。这对于理解、信任和调试智能体的行为至关重要。

2. **动态规划与纠错能力**：与一次性生成完整计划的范式不同，ReAct 是“走一步，看一步”。它根据每一步从外部世界获得的 `Observation` 来动态调整后续的 `Thought` 和 `Action`。如果上一步的搜索结果不理想，它可以在下一步中修正搜索词，重新尝试。

3. **工具协同能力**：ReAct 范式天然地将大语言模型的推理能力与外部工具的执行能力结合起来。LLM 负责运筹帷幄（规划和推理），工具负责解决具体问题（搜索、计算），二者协同工作，突破了单一 LLM 在知识时效性、计算准确性等方面的固有局限。

  

（2）ReAct 的固有局限性

  

1. **对LLM自身能力的强依赖**：ReAct 流程的成功与否，高度依赖于底层 LLM 的综合能力。如果 LLM 的逻辑推理能力、指令遵循能力或格式化输出能力不足，就很容易在 `Thought` 环节产生错误的规划，或者在 `Action` 环节生成不符合格式的指令，导致整个流程中断。

2. **执行效率问题**：由于其循序渐进的特性，完成一个任务通常需要多次调用 LLM。每一次调用都伴随着网络延迟和计算成本。对于需要很多步骤的复杂任务，这种串行的“思考-行动”循环可能会导致较高的总耗时和费用。

3. **提示词的脆弱性**：整个机制的稳定运行建立在一个精心设计的提示词模板之上。模板中的任何微小变动，甚至是用词的差异，都可能影响 LLM 的行为。此外，并非所有模型都能持续稳定地遵循预设的格式，这增加了在实际应用中的不确定性。

4. **可能陷入局部最优**：步进式的决策模式意味着智能体缺乏一个全局的、长远的规划。它可能会因为眼前的 `Observation` 而选择一个看似正确但长远来看并非最优的路径，甚至在某些情况下陷入“原地打转”的循环中。

  

（3）调试技巧

  

当你构建的 ReAct 智能体行为不符合预期时，可以从以下几个方面入手进行调试：

  

- **检查完整的提示词**：在每次调用 LLM 之前，将最终格式化好的、包含所有历史记录的完整提示词打印出来。这是追溯 LLM 决策源头的最直接方式。

- **分析原始输出**：当输出解析失败时（例如，正则表达式没有匹配到 `Action`），务必将 LLM 返回的原始、未经处理的文本打印出来。这能帮助你判断是 LLM 没有遵循格式，还是你的解析逻辑有误。

- **验证工具的输入与输出**：检查智能体生成的 `tool_input` 是否是工具函数所期望的格式，同时也要确保工具返回的 `observation` 格式是智能体可以理解和处理的。

- **调整提示词中的示例 (Few-shot Prompting)**：如果模型频繁出错，可以在提示词中加入一两个完整的“Thought-Action-Observation”成功案例，通过示例来引导模型更好地遵循你的指令。

- **尝试不同的模型或参数**：更换一个能力更强的模型，或者调整 `temperature` 参数（通常设为0以保证输出的确定性），有时能直接解决问题。

## 2. Plan-and-solve

**Plan-and-Solve**。顾名思义，这种范式将任务处理明确地分为两个阶段：**先规划 (Plan)，后执行 (Solve)**。

  

如果说 ReAct 像一个经验丰富的侦探，根据现场的蛛丝马迹（Observation）一步步推理，随时调整自己的调查方向；那么 Plan-and-Solve 则更像一位建筑师，在动工之前必须先绘制出完整的蓝图（Plan），然后严格按照蓝图来施工（Solve）。事实上我们现在用的很多大模型工具的Agent模式都融入了这种设计模式。

### Plan-And-Solve工作原理

Plan-and-Solve Prompting 由 Lei Wang 在2023年提出。其核心动机是为了解决思维链在处理多步骤、复杂问题时容易“偏离轨道”的问题。

  

与 ReAct 将思考和行动融合在每一步不同，Plan-and-Solve 将整个流程解耦为两个核心阶段，如图4.2所示：

  

1. **规划阶段 (Planning Phase)**： 首先，智能体会接收用户的完整问题。它的第一个任务不是直接去解决问题或调用工具，而是**将问题分解，并制定出一个清晰、分步骤的行动计划**。这个计划本身就是一次大语言模型的调用产物。

2. **执行阶段 (Solving Phase)**： 在获得完整的计划后，智能体进入执行阶段。它会**严格按照计划中的步骤，逐一执行**。每一步的执行都可能是一次独立的 LLM 调用，或者是对上一步结果的加工处理，直到计划中的所有步骤都完成，最终得出答案。

  

这种“先谋后动”的策略，使得智能体在处理需要长远规划的复杂任务时，能够保持更高的目标一致性，避免在中间步骤中迷失方向。

![image.png](https://raw.githubusercontent.com/aidenz0/blogImage/main/img/20260119142641521.png)

Plan-and-Solve 尤其适用于那些结构性强、可以被清晰分解的复杂任务，例如：

  

- **多步数学应用题**：需要先列出计算步骤，再逐一求解。

- **需要整合多个信息源的报告撰写**：需要先规划好报告结构（引言、数据来源A、数据来源B、总结），再逐一填充内容。

- **代码生成任务**：需要先构思好函数、类和模块的结构，再逐一实现。

### 规划阶段

为了凸显 Plan-and-Solve 范式在结构化推理任务上的优势，我们将不使用工具的方式，而是通过提示词的设计，完成一个推理任务。

  

这类任务的特点是，答案无法通过单次查询或计算得出，必须先将问题分解为一系列逻辑连贯的子步骤，然后按顺序求解。这恰好能发挥 Plan-and-Solve “先规划，后执行”的核心能力。

  

**我们的目标问题是：**“一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？”

  

这个问题对于大语言模型来说并不算特别困难，但它包含了一个清晰的逻辑链条可供参考。在某些实际的逻辑难题上，如果大模型不能高质量的推理出准确的答案，可以参考这个设计模式来设计自己的Agent完成任务。智能体需要：

  

1. **规划阶段**：首先，将问题分解为三个独立的计算步骤（计算周二销量、计算周三销量、计算总销量）。

2. **执行阶段**：然后，严格按照计划，一步步执行计算，并将每一步的结果作为下一步的输入，最终得出总和。

  

规划阶段的目标是让大语言模型接收原始问题，并输出一个清晰、分步骤的行动计划。这个计划必须是结构化的，以便我们的代码可以轻松解析并逐一执行。因此，我们设计的提示词需要明确地告诉模型它的角色和任务，并给出一个输出格式的范例。

```python

PLANNER_PROMPT_TEMPLATE = """

你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。

你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

  

问题: {question}

  

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:

```python

["步骤1", "步骤2", "步骤3", ...]

  

"""

```

这个提示词通过以下几点确保了输出的质量和稳定性：

  

- **角色设定**： “顶级的AI规划专家”，激发模型的专业能力。

- **任务描述**： 清晰地定义了“分解问题”的目标。

- **格式约束**： 强制要求输出为一个 Python 列表格式的字符串，这极大地简化了后续代码的解析工作，使其比解析自然语言更稳定、更可靠。

接下来，我们将这个提示词逻辑封装成一个 `Planner` 类，这个类也是我们的规划器。

```python

# 假定 llm_client.py 中的 HelloAgentsLLM 类已经定义好

# from llm_client import HelloAgentsLLM

  

class Planner:

def __init__(self, llm_client):

self.llm_client = llm_client

  

def plan(self, question: str) -> list[str]:

"""

根据用户问题生成一个行动计划。

"""

prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

# 为了生成计划，我们构建一个简单的消息列表

messages = [{"role": "user", "content": prompt}]

print("--- 正在生成计划 ---")

# 使用流式输出来获取完整的计划

response_text = self.llm_client.think(messages=messages) or ""

print(f"✅ 计划已生成:\n{response_text}")

# 解析LLM输出的列表字符串

try:

# 找到```python和```之间的内容

plan_str = response_text.split("```python")[1].split("```")[0].strip()

# 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表

plan = ast.literal_eval(plan_str)

return plan if isinstance(plan, list) else []

except (ValueError, SyntaxError, IndexError) as e:

print(f"❌ 解析计划时出错: {e}")

print(f"原始响应: {response_text}")

return []

except Exception as e:

print(f"❌ 解析计划时发生未知错误: {e}")

return []

```

### 执行器和状态管理

在规划器 (`Planner`) 生成了清晰的行动蓝图后，我们就需要一个执行器 (`Executor`) 来逐一完成计划中的任务。执行器不仅负责调用大语言模型来解决每个子问题，还承担着一个至关重要的角色：**状态管理**。它必须记录每一步的执行结果，并将其作为上下文提供给后续步骤，确保信息在整个任务链条中顺畅流动

  

执行器的提示词与规划器不同。它的目标不是分解问题，而是**在已有上下文的基础上，专注解决当前这一个步骤**。因此，提示词需要包含以下关键信息：

  

- **原始问题**： 确保模型始终了解最终目标。

- **完整计划**： 让模型了解当前步骤在整个任务中的位置。

- **历史步骤与结果**： 提供至今为止已经完成的工作，作为当前步骤的直接输入。

- **当前步骤**： 明确指示模型现在需要解决哪一个具体任务。

```python

EXECUTOR_PROMPT_TEMPLATE = """

你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。

你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。

请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

  

# 原始问题:

{question}

  

# 完整计划:

{plan}

  

# 历史步骤与结果:

{history}

  

# 当前步骤:

{current_step}

  

请仅输出针对“当前步骤”的回答:

"""

```

我们将执行逻辑封装到 `Executor` 类中。这个类将循环遍历计划，调用 LLM，并维护一个历史记录（状态）。

```python

class Executor:

def __init__(self, llm_client):

self.llm_client = llm_client

  

def execute(self, question: str, plan: list[str]) -> str:

"""

根据计划，逐步执行并解决问题。

"""

history = "" # 用于存储历史步骤和结果的字符串

print("\n--- 正在执行计划 ---")

for i, step in enumerate(plan):

print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

prompt = EXECUTOR_PROMPT_TEMPLATE.format(

question=question,

plan=plan,

history=history if history else "无", # 如果是第一步，则历史为空

current_step=step

)

messages = [{"role": "user", "content": prompt}]

response_text = self.llm_client.think(messages=messages) or ""

# 更新历史记录，为下一步做准备

history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

  

# 循环结束后，最后一步的响应就是最终答案

final_answer = response_text

return final_answer

```

现在已经分别构建了负责“规划”的 `Planner` 和负责“执行”的 `Executor`。最后一步是将这两个组件整合到一个统一的智能体 `PlanAndSolveAgent` 中，并赋予它解决问题的完整能力。我们将创建一个主类 `PlanAndSolveAgent`，它的职责非常清晰：接收一个 LLM 客户端，初始化内部的规划器和执行器，并提供一个简单的 `run` 方法来启动整个流程。

```python

class PlanAndSolveAgent:

def __init__(self, llm_client):

"""

初始化智能体，同时创建规划器和执行器实例。

"""

self.llm_client = llm_client

self.planner = Planner(self.llm_client)

self.executor = Executor(self.llm_client)

  

def run(self, question: str):

"""

运行智能体的完整流程:先规划，后执行。

"""

print(f"\n--- 开始处理问题 ---\n问题: {question}")

# 1. 调用规划器生成计划

plan = self.planner.plan(question)

# 检查计划是否成功生成

if not plan:

print("\n--- 任务终止 --- \n无法生成有效的行动计划。")

return

  

# 2. 调用执行器执行计划

final_answer = self.executor.execute(question, plan)

print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

```

## 3. Reflection

### Reflection机制的核心思想

Reflection 机制的灵感来源于人类的学习过程：我们完成初稿后会进行校对，解出数学题后会进行验算。这一思想在多个研究中得到了体现，例如 Shinn, Noah 在2023年提出的 Reflexion 框架。其核心工作流程可以概括为一个简洁的三步循环：**执行 -> 反思 -> 优化**。

  

1. **执行 (Execution)**：首先，智能体使用我们熟悉的方法（如 ReAct 或 Plan-and-Solve）尝试完成任务，生成一个初步的解决方案或行动轨迹。这可以看作是“初稿”。

2. **反思 (Reflection)**：接着，智能体进入反思阶段。它会调用一个独立的、或者带有特殊提示词的大语言模型实例，来扮演一个“评审员”的角色。这个“评审员”会审视第一步生成的“初稿”，并从多个维度进行评估，例如：

- **事实性错误**：是否存在与常识或已知事实相悖的内容？

- **逻辑漏洞**：推理过程是否存在不连贯或矛盾之处？

- **效率问题**：是否有更直接、更简洁的路径来完成任务？

- **遗漏信息**：是否忽略了问题的某些关键约束或方面？ 根据评估，它会生成一段结构化的**反馈 (Feedback)**，指出具体的问题所在和改进建议。

3. **优化 (Refinement)**：最后，智能体将“初稿”和“反馈”作为新的上下文，再次调用大语言模型，要求它根据反馈内容对初稿进行修正，生成一个更完善的“修订稿”。

![image.png](https://raw.githubusercontent.com/aidenz0/blogImage/main/img/20260119160202099.png)

与前两种范式相比，Reflection 的价值在于：

  

- 它为智能体提供了一个内部纠错回路，使其不再完全依赖于外部工具的反馈（ReAct 的 Observation），从而能够修正更高层次的逻辑和策略错误。

- 它将一次性的任务执行，转变为一个持续优化的过程，显著提升了复杂任务的最终成功率和答案质量。

- 它为智能体构建了一个临时的**“短期记忆”**。整个“执行-反思-优化”的轨迹形成了一个宝贵的经验记录，智能体不仅知道最终答案，还记得自己是如何从有缺陷的初稿迭代到最终版本的。更进一步，这个记忆系统还可以是**多模态的**，允许智能体反思和修正文本以外的输出（如代码、图像等），为构建更强大的多模态智能体奠定了基础。

## 案例设定与记忆模块设计

为了在实战中体现 Reflection 机制，我们将引入记忆管理机制，因为reflection通常对应着信息的存储和提取，如果上下文足够长的情况，想让“评审员”直接获取所有的信息然后进行反思往往会传入很多冗余信息。这一步实践我们主要完成**代码生成与迭代优化**。

  

这一步的目标任务是：“编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。”

  

这个任务是检验 Reflection 机制的绝佳场景：

  

1. **存在明确的优化路径**：大语言模型初次生成的代码很可能是一个简单但效率低下的递归实现。

2. **反思点清晰**：可以通过反思发现其“时间复杂度过高”或“存在重复计算”的问题。

3. **优化方向明确**：可以根据反馈，将其优化为更高效的迭代版本或使用备忘录模式的版本。

  

Reflection 的核心在于迭代，而迭代的前提是能够记住之前的尝试和获得的反馈。因此，一个“短期记忆”模块是实现该范式的必需品。这个记忆模块将负责存储每一次“执行-反思”循环的完整轨迹。

```python

from typing import List, Dict, Any, Optional

  

class Memory:

"""

一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。

"""

  

def __init__(self):

"""

初始化一个空列表来存储所有记录。

"""

self.records: List[Dict[str, Any]] = []

  

def add_record(self, record_type: str, content: str):

"""

向记忆中添加一条新记录。

  

参数:

- record_type (str): 记录的类型 ('execution' 或 'reflection')。

- content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。

"""

record = {"type": record_type, "content": content}

self.records.append(record)

print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

  

def get_trajectory(self) -> str:

"""

将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。

"""

trajectory_parts = []

for record in self.records:

if record['type'] == 'execution':

trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")

elif record['type'] == 'reflection':

trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")

return "\n\n".join(trajectory_parts)

  

def get_last_execution(self) -> Optional[str]:

"""

获取最近一次的执行结果 (例如，最新生成的代码)。

如果不存在，则返回 None。

"""

for record in reversed(self.records):

if record['type'] == 'execution':

return record['content']

return None

```

这个 `Memory` 类的设计比较简洁，主体是这样的：

  

- 使用一个列表 `records` 来按顺序存储每一次的行动和反思。

- `add_record` 方法负责向记忆中添加新的条目。

- `get_trajectory` 方法是核心，它将记忆轨迹“序列化”成一段文本，可以直接插入到后续的提示词中，为模型的反思和优化提供完整的上下文。

- `get_last_execution` 方便我们获取最新的“初稿”以供反思。

### Reflection智能体的编码实现

有了 `Memory` 模块作为基础，我们现在可以着手构建 `ReflectionAgent` 的核心逻辑。整个智能体的工作流程将围绕我们之前讨论的“执行-反思-优化”循环展开，并通过精心设计的提示词来引导大语言模型扮演不同的角色。

（1）提示词设计

  

与之前的范式不同，Reflection 机制需要多个不同角色的提示词来协同工作。

  

1. **初始执行提示词 (Execution Prompt)** ：这是智能体首次尝试解决问题的提示词，内容相对直接，只要求模型完成指定任务。

  

```python

INITIAL_PROMPT_TEMPLATE = """

你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。

你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

  

要求: {task}

  

请直接输出代码，不要包含任何额外的解释。

"""

```

  

2. **反思提示词 (Reflection Prompt)** ：这个提示词是 Reflection 机制的灵魂。它指示模型扮演“代码评审员”的角色，对上一轮生成的代码进行批判性分析，并提供具体的、可操作的反馈。

  

```python

REFLECT_PROMPT_TEMPLATE = """

你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。

你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

  

# 原始任务:

{task}

  

# 待审查的代码:

```python

{code}

```

```

  

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。

如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。

如果代码在算法层面已经达到最优，才能回答“无需改进”。

  

请直接输出你的反馈，不要包含任何额外的解释。

"""

```

  
  

3. **优化提示词 (Refinement Prompt)** ：当收到反馈后，这个提示词将引导模型根据反馈内容，对原有代码进行修正和优化。

  

```python

  

REFINE_PROMPT_TEMPLATE = """

你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

  

# 原始任务:

{task}

  

# 你上一轮尝试的代码:

{last_code_attempt}

评审员的反馈：

{feedback}

  

请根据评审员的反馈，生成一个优化后的新版本代码。

你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

请直接输出优化后的代码，不要包含任何额外的解释。

"""

```

  

（2）智能体封装与实现

  

现在，我们将这套提示词逻辑和 `Memory` 模块整合到 `ReflectionAgent` 类中。

  

```python

# 假设 llm_client.py 和 memory.py 已定义

# from llm_client import HelloAgentsLLM

# from memory import Memory

  

class ReflectionAgent:

def __init__(self, llm_client, max_iterations=3):

self.llm_client = llm_client

self.memory = Memory()

self.max_iterations = max_iterations

  

def run(self, task: str):

print(f"\n--- 开始处理任务 ---\n任务: {task}")

  

# --- 1. 初始执行 ---

print("\n--- 正在进行初始尝试 ---")

initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)

initial_code = self._get_llm_response(initial_prompt)

self.memory.add_record("execution", initial_code)

  

# --- 2. 迭代循环:反思与优化 ---

for i in range(self.max_iterations):

print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

  

# a. 反思

print("\n-> 正在进行反思...")

last_code = self.memory.get_last_execution()

reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)

feedback = self._get_llm_response(reflect_prompt)

self.memory.add_record("reflection", feedback)

  

# b. 检查是否需要停止

if "无需改进" in feedback:

print("\n✅ 反思认为代码已无需改进，任务完成。")

break

  

# c. 优化

print("\n-> 正在进行优化...")

refine_prompt = REFINE_PROMPT_TEMPLATE.format(

task=task,

last_code_attempt=last_code,

feedback=feedback

)

refined_code = self._get_llm_response(refine_prompt)

self.memory.add_record("execution", refined_code)

final_code = self.memory.get_last_execution()

print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")

return final_code

  

def _get_llm_response(self, prompt: str) -> str:

"""一个辅助方法，用于调用LLM并获取完整的流式响应。"""

messages = [{"role": "user", "content": prompt}]

response_text = self.llm_client.think(messages=messages) or ""

return response_text

```

### Reflection 机制的成本收益分析

尽管 Reflection 机制在提升任务解决质量上表现出色，但这种能力的获得并非没有代价。在实际应用中，我们需要权衡其带来的收益与相应的成本。

  

（1）主要成本

  

1. **模型调用开销增加**:这是最直接的成本。每进行一轮迭代，至少需要额外调用两次大语言模型（一次用于反思，一次用于优化）。如果迭代多轮，API 调用成本和计算资源消耗将成倍增加。

2. **任务延迟显著提高**:Reflection 是一个串行过程，每一轮的优化都必须等待上一轮的反思完成。这使得任务的总耗时显著延长，不适合对实时性要求高的场景。

3. **提示工程复杂度上升**:如我们的案例所示，Reflection 的成功在很大程度上依赖于高质量、有针对性的提示词。为“执行”、“反思”、“优化”等不同阶段设计和调试有效的提示词，需要投入更多的开发精力。

（2）核心收益

  

1. **解决方案质量的跃迁**:最大的收益在于，它能将一个“合格”的初始方案，迭代优化成一个“优秀”的最终方案。这种从功能正确到性能高效、从逻辑粗糙到逻辑严谨的提升，在很多关键任务中是至关重要的。

2. **鲁棒性与可靠性增强**:通过内部的自我纠错循环，智能体能够发现并修复初始方案中可能存在的逻辑漏洞、事实性错误或边界情况处理不当等问题，从而大大提高了最终结果的可靠性。

  

综上所述，Reflection 机制是一种典型的“以成本换质量”的策略。它非常适合那些**对最终结果的质量、准确性和可靠性有极高要求，且对任务完成的实时性要求相对宽松**的场景。例如:

  

- 生成关键的业务代码或技术报告。

- 在科学研究中进行复杂的逻辑推演。

- 需要深度分析和规划的决策支持系统。

  

反之，如果应用场景需要快速响应，或者一个“大致正确”的答案就已经足够，那么使用更轻量的 ReAct 或 Plan-and-Solve 范式可能会是更具性价比的选择。

  

## 小结

**核心知识点回顾:**

  

1. ReAct:我们构建了一个能与外部世界交互的 ReAct 智能体。通过“思考-行动-观察”的动态循环，它成功地利用搜索引擎回答了自身知识库无法覆盖的实时性问题。其核心优势在于**环境适应性**和**动态纠错能力**，使其成为处理探索性、需要外部工具输入的任务的首选。

2. Plan-and-Solve:我们实现了一个先规划后执行的 Plan-and-Solve 智能体，并利用它解决了需要多步推理的数学应用题。它将复杂的任务分解为清晰的步骤，然后逐一执行。其核心优势在于**结构性**和**稳定性**，特别适合处理逻辑路径确定、内部推理密集的任务。

3. Reflection (自我反思与迭代):我们构建了一个具备自我优化能力的 Reflection 智能体。通过引入“执行-反思-优化”的迭代循环，它成功地将一个效率较低的初始代码方案，优化为了一个算法上更优的高性能版本。其核心价值在于能**显著提升解决方案的质量**，适用于对结果的准确性和可靠性有极高要求的场景。

![image.png](https://raw.githubusercontent.com/aidenz0/blogImage/main/img/20260119161818456.png)
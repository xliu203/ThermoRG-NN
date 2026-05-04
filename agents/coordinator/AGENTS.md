# AGENTS — 工作规范与调度规则

## Startup 强制条款（必须遵守）

每次 spawn 进入**新 session** 时（由 `/new` 或 `/reset` 触发），**第一条 ToolCall** 必须是按顺序读取以下三个文件（只使用 read tool，不得先执行其他任何操作）：

1. `/home/node/.openclaw/workspace/agents/coordinator/SOUL.md` — 理解你的核心人格
2. `/home/node/.openclaw/workspace/agents/coordinator/SKILL.md` — 理解你的技能和工具
3. `/home/node/.openclaw/workspace/agents/coordinator/AGENTS.md` — 理解你的工作规范

完成之前不得调用 edit/exec/write/search/其他任何工具。

**重要澄清：**
- 若 sub-agent 在**已有 ongoing session** 中被再次派发任务（同一 session 内的后续任务），**不需要**重复读取 startup 文件。
- 冷启动：每个 agent 的第一个任务都要走 startup → 保证人格和行为规范被认知。
- 热复用：session 内任务保持连贯性，不重复打断 agent 的记忆和状态。

## Session Startup（每次启动必须执行）

1. 读 `SOUL.md`（本目录）— 理解你的核心人格
2. 读 `SKILL.md`（本目录）— 理解你的技能和工具
3. 读 `AGENTS.md`（本目录）— 理解你的调度职责和工作流
4. 读项目相关文件 — 理解当前任务上下文
5. 读 `memory/YYYY-MM-DD.md`（今天 + 昨天）— 最近发生了什么

## 调度规则

| 任务类型 | 指派目标 | 模型 |
|---------|----------|------|
| 代码编写、调试、重构 | Coding Agent | deepseek/deepseek-v4-flash |
| 论文润色、学术写作 | Paper Editor | deepseek/deepseek-v4-pro |
| 理论推导、概念辨析 | Theory Agent | deepseek/deepseek-v4-pro |
| 内容整合、格式清洗 | Content Writer | deepseek/deepseek-v4-flash |

## Spawn 模型指定（必须遵守）

每次 `sessions_spawn` 必须显式指定 `model` 参数，禁止依赖 session 默认值。

正确示例：
```python
sessions_spawn(
    label="coding-fix",
    mode="run",
    runtime="subagent",
    model="deepseek/deepseek-v4-flash",  # ← 必须显式传入
    task="..."
)
```

错误（未指定 model）：
```python
sessions_spawn(label="coding-fix", mode="run", runtime="subagent", task="...")  # ← 会用 session 默认模型
```

**原因**：`SKILL.md` 的 `Model:` 字段只是描述，`sessions_spawn` 工具不自动读该字段推断 model，必须显式传参。

## 默认工作流

1. 理解用户意图，若需求模糊则主动提问。
2. 拆解任务，生成明确、可并行的子任务列表。
3. 将每个子任务交给最匹配的专家 Agent，并行分发。
4. 收集所有专家结果，整合并检查一致性。
5. 向用户交付最终结果，并说明改动点。

# AGENTS — 工作规范

## 技术栈
- 前端: TypeScript, React, Next.js, Tailwind CSS
- 后端: Node.js, Python, Go
- 数据库: PostgreSQL, SQLite, Redis
- 工具: Git, Docker, CI/CD

## 工作流
1. 理解需求，若有歧义主动追问
2. 设计方案（架构图、数据流、接口定义）
3. 编写代码，遵循项目现有代码风格
4. 编写/更新测试
5. 交付代码并说明改动点

## Startup 强制条款（必须遵守）

每次 spawn 进入新 session 时，**第一条 ToolCall** 必须是按顺序读取以下文件（只使用 read tool，不得先执行其他任何操作）：

1. `SOUL.md`（本目录）— 理解你的核心人格
2. `SKILL.md`（本目录）— 理解你的技能和工具
3. `AGENTS.md`（本目录）— 理解你的工作规范

完成之前不得调用 edit/exec/write/search/其他任何工具。

## Session Startup（每次启动必须执行）

1. 读 `SOUL.md`（本目录）— 理解你的核心人格
2. 读 `SKILL.md`（本目录）— 理解你的技能和工具
3. 读 `AGENTS.md`（本目录）— 理解你的工作规范
4. 读项目相关的 `phase1_experiments/` 代码 — 理解当前任务上下文
5. 读 `memory/YYYY-MM-DD.md`（今天 + 昨天）— 最近发生了什么

## 输出规范
- 提供完整可运行的代码片段或文件
- 标明需要手动执行的操作（如安装依赖、迁移数据库）
- 必要时附上执行命令

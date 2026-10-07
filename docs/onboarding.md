# 安装后的知识库接入引导

## 用户流程

安装 Agent 安装依赖并运行 install → 检查状态 → 在当前对话询问目录 → 预览范围 → 用户确认范围 → 保存配置与索引 → 试搜验证。

安装步骤：

```powershell
uv sync --locked --extra mcp
uv run --locked --extra mcp knowledge-rag install
```

install 注册/复用当前 Codex 服务，启动真实 MCP 子进程验证工具清单并调用 get_setup_status。
返回 `setup.question` 与 `agent_instruction`；安装 Agent 应直接在当前对话提出问题并等待回答。
示例知识库存在时仍为 needs_knowledge_base，不会误认为用户已经接入真实资料。
用户可选择稍后配置；此时安装已完成，本轮不再追问。

此引导通过安装任务指令、安装结果和 MCP 服务 instructions 配合实现；服务不会自行发起模型回合。
终端直接运行 install 会输出结构化下一步，聊天体验由负责安装的 Agent 执行。
其他宿主连接成功后调用 get_setup_status 即可使用相同流程；不依赖 elicitation 表单支持。

## 状态与工具

- needs_knowledge_base：仅有随项目提供的演示库，需要询问用户目录。
- needs_index：已明确配置真实来源，但尚无已索引文档；按返回配置建立索引。
- ready：真实来源已有索引，可直接检索。

preview_knowledge_base 接收 kb_id、root 和可选 include/exclude。
root 必须是用户提供的绝对目录，默认匹配 Markdown 并排除隐藏路径。
返回文件数、字节数、规则、最多10个路径样例、超大文件情况和 preview_id；仅统计文件元数据，不读取正文。
目录不存在、空范围、超大文件或来源身份冲突时，不进入接入步骤。

connect_knowledge_base 使用一致参数、preview_id 与 confirmed=true。
confirmed 表示 Agent 已取得用户对该范围的确认，并非由服务自动推断同意。
预览标识覆盖目录、筛选规则、配置与文件大小/时间/清单，变化后须重新预览。
索引后再次核对范围，防止接入期间新增的文件未经展示即发布。
预览不使用正文哈希；这是范围预览，不是内容版本锁。最终文档版本由索引 source_hash 标识。

## 配置与恢复

新配置保存到原数据库同目录的 `<原数据库名>.connected.toml`，所有路径为绝对路径。
每次接入先建立独立候选数据库，核对文档可读后原子替换配置指针。索引/发布失败时继续使用原配置。
同一个服务进程后续请求会读取新配置；重新启动也会读取它。CLI index/list/search/read 同样使用保存的活动配置。
默认源文件配置不被改写；旧数据库快照保留在本地状态目录，当前没有自动清理功能。
benchmark 始终使用明确传入的源文件配置，避免个人来源意外进入演示评测。

当前接入操作只在单个服务进程内串行，不支持多个配置写入进程同时接入同一配置。
来源目录确认并非多用户授权机制；这是可信本地服务的能力。

## 可量化记录

setup_status、setup_preview、setup_connect、install_codex 均有 trace_id。
预览记录文件数与字节数，接入记录索引/验证/发布耗时和文档数，安装记录注册与协议验证耗时。
来源路径在 trace 中仅记哈希，预览与返回配置路径供用户/Agent 查看；正文不进入引导日志。
目录被读取、配置保存、同一进程立即生效和失败保留原配置都有工程测试。
模型是否主动提问是 Agent 行为，需要宿主端到端验证，当前工程测试验证返回问题与服务指令。

## CLI 备用流程

```powershell
uv run --locked knowledge-rag setup-status
uv run --locked knowledge-rag preview personal "D:/YourKnowledge"
# 展示范围并取得用户确认后：
uv run --locked knowledge-rag connect personal "D:/YourKnowledge" --preview-id <returned-id> --confirmed
uv run --locked knowledge-rag search "用户提供的样例问题" --kb personal
```

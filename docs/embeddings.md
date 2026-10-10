# Embedding 与向量检索使用

支持本地 ONNX CPU 编码和兼容 `/embeddings` 的 HTTP API。BM25仍是默认策略。
本地模型与缓存默认位于项目下忽略的 `.state/`，项目位于哪个盘，文件就落在哪个盘。
模型不会在查询时自动下载；服务重启后的首次 dense 查询需要加载模型，后续复用。

## 本地模型

```powershell
$env:UV_CACHE_DIR = "$PWD/.state/uv-cache"
$env:TMP = "$PWD/.state/tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
uv sync --locked --extra mcp --extra vector --extra embedding-local
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml model-download
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml embedding-check
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml vector-build
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml vector-status
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml search "怎样找到意思相近的资料" --kb learning --strategy dense
```

上述配置只接入公开演示资料。首次模型下载固定官方 E5-small commit 与两个大文件 SHA-256，
约487 MB，包括约470 MB ONNX权重与约17 MB分词器；不重复下载 PyTorch 权重。
依赖、uv 缓存、临时目录可按示例放在项目盘，已存在的 Python 安装不会被迁移。
来源：[官方模型](https://huggingface.co/intfloat/multilingual-e5-small)、[ONNX CPU运行接口](https://onnxruntime.ai/docs/api/python/api_summary.html)。

本地实现使用模型 tokenizer、`query: `/`passage: `前缀、attention-mask mean pooling 和归一化。
只支持输出 token hidden states 的兼容 ONNX 模型；其他模型的 pooling/输入规则需另写适配，不能直接换模型名称。
接入自备模型时配置 `model_dir`，目录至少提供 `onnx/model.onnx`、`tokenizer.json`，并准确设置维度、最大token与模板。
模型和分词器文件内容指纹参与缓存与索引契约，编码不兼容时明确拒绝；不要用未校验的模型混入现有索引。

## API 配置

复制 `config.api.example.toml` 为被忽略的 `config.api.local.toml`，设置：

- `provider = "api"`、服务 `base_url`（例如兼容服务的 `/v1` 根路径，程序追加 `/embeddings`）。
- 服务支持的 `model`、`dimension`、`max_tokens` 和模型版本标识 `revision`。
- 适配该模型的 tiktoken encoding 名称 `tokenizer`；示例 `cl100k_base` 不能默认适用于所有服务。
- `template = "plain-v1"` 或模型要求的 E5 前缀模板。
- `api_key_env` 只填写环境变量名称，不填写密钥值。

```powershell
uv sync --locked --extra mcp --extra vector --extra embedding-api
# 在本地终端安全设置 KNOWLEDGE_RAG_EMBEDDING_KEY；不要把密钥写入仓库或对话。
uv run --locked --extra vector --extra embedding-api knowledge-rag --config config.api.local.toml vector-build
uv run --locked --extra vector --extra embedding-api knowledge-rag --config config.api.local.toml search "你的问题" --strategy dense
```

当前协议发送 model/input/encoding_format，按响应 index 恢复顺序，校验数量/维度/有限数并归一化。
不支持所有供应商专有的鉴权和请求格式；需要额外参数或非 tiktoken 分词器的服务应增加独立适配。
远程 token 计数和模型 revision 以配置及服务契约为准：客户端计数不是计费凭证，不能保证服务端没有内部截断。
应选择明确支持该输入长度及不静默截断的服务。模型版本别名变化时重新建立版本标识与索引。
仅允许 HTTPS，或回环地址 HTTP；不自动跟随重定向，不把响应正文、输入或密钥写进错误/trace。
选择 API 后，待编码片段和问题会发送到所配置服务；仅 BM25 搜索不调用 Embedding API。
未提供真实API凭证前，验收范围是模拟响应的协议、错误、归一化和脱敏，不代表供应商可用性。

## 切换与构建

在已有私有接入配置中加入或修改 `[embedding]`、`[vector]`，然后运行 `vector-build`。
修改 provider、模型、维度、revision、token模板会改变契约；dense拒绝混用，不静默回退BM25。
密钥值不参与配置序列化或日志。缓存按模型契约、角色和实际输入文本区分。
`query_cache = false` 为默认值，方便观察实际查询编码耗时；可显式启用相同问题的缓存。

构建先创建独立 SQLite 快照与 Qdrant 候选库，验证点数、来源版本、完整窗口，再原子替换共同的活动指针。
成功后 BM25、dense、原文读取使用同一份 SQLite快照；失败保留上一代次，失败报告在候选目录中。
`index` 在明确配置 Embedding/Vector 后等同构建新代次；没有这些配置时保留原词法索引行为。
源文件变化后需重新构建，查询读取上次发布快照；`vector-status` 会检查来源变化。
无变化输入从缓存复用，但仍生成候选快照与向量库，不是零磁盘写入。旧代次暂不自动清理。
所有资料删除时，可发布0点的新代次；接入/范围改变后旧指针不再匹配，应重新构建。

文档按完整字符覆盖生成token窗口，每个窗口计算完整前缀、标题、heading与special tokens预算。
不静默截断；标题超限或查询超限明确报错。窗口命中去重到原父片段，沿用原引用与正文字符预算。
长父片段的结果摘要仍从片段开头截取；命中窗口的局部证据摘取属于后续上下文优化。

local模式只支持精确扫描，并独占当前代次目录；进程内查询串行化。
多个MCP进程共享使用Server模式：配置`mode = "server"`、`url`与`api_key_env`，构建使用独立collection。
Server模式运行、并发、旧代次清理与生产部署仍需单独验收。

## MCP 与评测

`search_knowledge` 新增 `strategy`：`bm25`/`overlap`/`dense`。已有调用不传参数时保持BM25。
`get_setup_status` 在配置向量模块后包含`vector_index`状态摘要。安装宿主若缓存旧工具schema，需要重新连接服务。
构建和模型下载由CLI完成，不让读取工具自动触发长时间构建。

```powershell
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml benchmark --strategy bm25
uv run --locked --extra vector --extra embedding-local knowledge-rag --config config.vector.example.toml benchmark --strategy dense
uv run --locked knowledge-rag compare <bm25-report.json> <dense-report.json>
```

benchmark 的构建目录与个人活动索引分离，记录模型契约、窗口规则、CPU线程/批量大小、查询缓存开关，
构建和查询编码指标、进程峰值工作集及阶段耗时。排名和正文必须重复一致，分数只允许明确的浮点容差。
效果须比较质量、无答案候选、范围泄漏和返回大小，不能只看Recall；相似度不是回答置信度。
真实问题与标签仍需独立人工评测，公共合成集不能证明个人知识库效果。

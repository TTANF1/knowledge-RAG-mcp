# 隐私与公开发布

## 本地数据与公开资料

| 内容 | 保存位置 | 提交策略 |
| --- | --- | --- |
| 数据库、活动配置、来源清单、接入验证、日常日志 | `.state/` | 忽略 |
| 真实查询集、个人实验、对话摘录、学习笔记、本机状态 | `.local/` 或 `benchmarks/private/` | 忽略 |
| 原始 benchmark 报告、快照、逐题结果和 trace | `benchmarks/runs/`、`benchmarks/baselines/` | 忽略 |
| 代码、通用文档、模板、人工演示语料 | `src/`、`docs/`、`examples/` | 可公开，需检查 |
| 批准的合成问题集 | `benchmarks/datasets/smoke-v1.jsonl` | 可公开 |
| 核验合成语料后导出的聚合指标 | `benchmarks/public/` | 可公开，需检查 |

JSONL 默认忽略，仅放行批准的演示问题集。数据库变体、日志、连接配置、私有环境变量和证书文件也默认忽略。
公开文档用占位路径，避免记录具体机器、用户名、个人文档、私有查询或引文。
本地继续积累完整证据与实验，用于改善 RAG；无需为了开源而删除这些资料。

## 提交与推送检查

```powershell
uv run --locked python scripts/check_publication.py
uv run --locked python scripts/check_publication.py --staged
uv run --locked python scripts/check_publication.py --history
```

检查工作区候选、暂存内容或所有可达旧提交中的私有路径、数据库/日志文件、本机绝对路径及明显密钥特征。
错误只显示文件、行号和类别，不输出匹配的私密内容。强制添加被忽略的文件也会被暂存检查拦截。
`.githooks/pre-commit` 和 `pre-push` 分别执行暂存检查、含历史检查；在无既有自定义钩子时可安装：

```powershell
git config --local core.hooksPath .githooks
```

已有钩子应合并调用检查脚本，避免覆盖原有行为。克隆后的用户需要自行启用钩子；此本地 Git 设置不会随仓库传播。
这是规则检查，不能识别全部语义隐私，也不能阻止用户显式绕过本地钩子；公开文档与新演示数据仍需审阅。

## 已被跟踪和旧历史

新增 gitignore 不会停止跟踪已有文件。取消跟踪只修改索引，保留本地副本；旧提交仍保存原内容。
推送检查扫描所有可达历史，发现旧本机信息时会阻止推送。
本项目不会为了公开而自动重写历史。首次发布可从当前已经清理的工作区生成全新副本：

```powershell
uv run --locked python scripts/export_public.py
```

输出目录位于忽略的 `.local/releases/`，只复制通过检查的 Git 文件候选，不复制 `.git`、私有目录或运行数据。
在输出目录建立全新的 Git 仓库后首次发布，便不会携带旧本机提交。输出不会修改源仓库或现有知识库连接。
源仓库中被取消跟踪的文件会显示为暂存删除，这是停止公开跟踪的结果；完整本地原始档案仍保留。

## 公开实验摘要

```powershell
uv run --locked python scripts/export_benchmark_summary.py <local-report.json> benchmarks/public/<new-experiment>
```

导出前核验报告的资料清单、内容哈希与问题集完全对应公开演示内容。
导出仅保留参数、聚合指标、分类指标、阶段时间和代码指纹；完整来源、逐题结果、引文与日志继续留在本地。
用户真实问题的实验结论默认也留在本地，需独立脱敏审阅后才能公开；不得伪装为演示语料绕过校验。

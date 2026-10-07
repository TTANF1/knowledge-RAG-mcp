# ADR-0002：向量存储接入方向

- 日期：2026-10-07
- 状态：proposed
- 阶段：规划；尚未安装向量依赖、下载模型或建立向量索引。

## 目标

引入真实语义检索能力，保持检索流程可理解、效果可测量，并服务多个 Agent 对同一知识库的访问。
原始资料、向量、缓存、报告和本机设置均保留本地。默认不向远程 Embedding API 发送资料。

## 推荐架构

- SQLite 继续负责原文快照、元数据、片段身份、行号和版本。
- Qdrant 负责向量的持久化、距离检索和候选范围过滤。
- EmbeddingProvider 独立负责文档/查询编码，模型和数据库可分别替换。
- BM25 保持既有对照，新增 dense；验证两路结果后再增加 hybrid，不同时加入重排。

运行时推荐本地 Qdrant Server。Qdrant 客户端的持久化 local 模式会独占存储目录，
不适合作为多个 MCP 进程和索引任务同时访问的正式路径；local/in-memory 模式仅用于隔离工程测试。
本地服务默认只绑定回环地址，Docker 持久化使用 named volume，服务镜像在实施时固定版本/摘要。
Docker 卷中的数据库同样属于私人资料，虽不在Git工作区，仍需纳入本机备份与清理策略。

## Embedding 起步模型

建议先用 `intfloat/multilingual-e5-small` 建立 CPU 可运行的中英混合基线：384维，512token限制。
模型要求查询使用 `query: ` 前缀，片段使用 `passage: ` 前缀；前缀、归一化和输入模板必须纳入版本。
使用 sentence-transformers 作为可选本地实现；不让仅使用 BM25 的安装被迫下载模型。
模型仓库 commit revision 与运行依赖在实施时固定，首次联网只下载模型；下载完成后可离线运行。

`BAAI/bge-m3` 作为后续候选：1024维、最长8192token、多语言。它需要不同资源配置和重建向量索引。
模型卡能力不等于本项目检索收益，是否替换由同一问题集的质量与开销对照决定。

## 关键约束

向量数据库和 SQLite 无跨库事务，采用候选代次构建与单一活动指针发布，而不是在检索中混用更新了一半的数据。
向量 payload 只含检索与一致性校验必要的身份/过滤字段，原文从 SQLite 按当前版本解析回。
不同模型、维度、模板或源快照不能共用一个索引身份；不兼容时明确提示重建。

## 官方依据

- [Qdrant 本地服务与 Windows 存储说明](https://qdrant.tech/documentation/quickstart/)
- [Qdrant local 模式的并发锁实现](https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/local/qdrant_local.py)
- [Multilingual E5 small 模型卡](https://huggingface.co/intfloat/multilingual-e5-small)
- [BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)

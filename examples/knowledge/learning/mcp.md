---
title: MCP 检索工具约定
status: verified
kind: learning
---
# MCP 检索工具约定

search_knowledge 返回证据片段；read_document 按需展开索引中的原文；list_knowledge_bases 列出可检索的知识库。

MCP 连接不保证 Agent 每次自动检索。宿主需要连接服务，Agent 需要在陈述过往事实前调用检索工具。

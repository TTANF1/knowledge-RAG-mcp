---
title: 检索评测指标
status: verified
kind: learning
---
# 检索评测指标

## Recall
Recall@k 衡量需要的证据中，有多少出现在前 k 个结果里。答案需要两份文档而只找回一份时，召回率为 0.5。

## 排名与性能
MRR 关注第一个相关结果的位置。p95 延迟衡量较慢一端的请求耗时，不能与平均延迟混为一谈。

## 上下文
字符数和 UTF-8 字节数不是模型 token 数。要声称节省 token，必须使用指定模型的 tokenizer 或实际 usage 数据。

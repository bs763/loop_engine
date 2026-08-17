# -*- coding: utf-8 -*-
"""LLM 层:可切换 provider + 13 机制族引导生成 / 审查精判。

文件:
  - provider.py    统一 LLMProvider 接口 + OpenAI 兼容实现(DeepSeek/GLM)+ Mock + 工厂
  - mechanisms.py  12 机制族数据(图表 7:8 时序 + 4 截面)+ 生成/审查 prompt + 表达式抽取/裁决

依赖 engine(算子/字段语法、表达式解析),不依赖回测指标(保持生成端 Goodhart 隔离)。
"""

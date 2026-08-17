# -*- coding: utf-8 -*-
"""核心算法层(纯 Python,与 Loop 编排层物理分离)。

模块(对应 docs/项目执行指南.md M1–M9):
  - operators.py    M1  14 算子(时序8 / 逐元素4 / 截面2)+ 注册表
  - expression.py   M1  表达式树(生成 / 解析 / 求值 / 哈希)
  - fsa.py          M8  频繁子树规避(骨架抽象 + 支持度禁止 + 变体上限)
  - review.py       M6  审查四过滤(截面退化 / 同质简化 / 跨量纲拒 / 最小复杂度)
  - perturb.py      M4  参数扰动(加权最小二乘梯度 + 动量 EMA + Adam 步长)
  - checkpoint.py   M9  检查点原子读写(断点续跑)
  - evolve.py       M3  五维演化引擎(变异/交叉/扰动/随机/LLM)
"""

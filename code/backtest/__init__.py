# -*- coding: utf-8 -*-
"""回测接口层:把表达式 / 因子面板 → 回测指标。

文件:
  - interface.py          FactorMetrics 契约 + 抽象 Evaluator
  - alphalab_adapter.py   真实适配器(调用户的 alphalab check)
  - mock.py               离线 Mock evaluator

回测引擎口径以用户 alphalab 为准(universe=中证全指、行业+市值中性、单边千一、2018-2025)。
"""

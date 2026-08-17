# -*- coding: utf-8 -*-
"""回测接口:FactorMetrics 契约 + 抽象 Evaluator。

面板约定(与 alphalab 输入一致):宽表,index=date(DatetimeIndex,name='date'),
columns=order_book_id(米筐格式),values=因子值(float)。

FactorMetrics 涵盖 M7 十一项过滤所需的全部字段(horizon 默认 5,对应 5 日换仓):
  IC 类:ic_mean / icir / icir_annual / t_stat_nw / positive_ratio
  多空:ls_annual / ls_sharpe / ls_max_dd / calmar(= ls_annual / |ls_max_dd|)
  多头超额:long_excess_annual / long_excess_sharpe
  按年:annual_ls_return {年: 多空收益}、annual_ic {年: ic_mean}(给「每年都过」)
  序列:ic_series(给 IC 相关性去重)、long_excess_nav(给滚动 9/12 月超额)
  入库:admission_pass(alphalab --gate,可选)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FactorMetrics:
    """单个因子在某 horizon 下的回测指标。"""
    expr: str = ""
    direction: int = 0          # alphalab 自动定方向:1 正向 / -1 反向
    horizon: int = 5

    # ---- IC 类 ----
    ic_mean: float = float("nan")
    icir: float = float("nan")
    icir_annual: float = float("nan")
    t_stat_nw: float = float("nan")
    positive_ratio: float = float("nan")

    # ---- 多空(long-short)----
    ls_annual: float = float("nan")
    ls_sharpe: float = float("nan")
    ls_max_dd: float = float("nan")
    calmar: float = float("nan")

    # ---- 多头超额(long-excess)----
    long_excess_annual: float = float("nan")
    long_excess_sharpe: float = float("nan")

    # ---- 分组单调性 ----
    monotonicity: float = float("nan")   # 1~10 分组收益单调程度 ∈ [0,1]

    # ---- 按年(给「每年都过」)----
    annual_ls_return: dict = field(default_factory=dict)   # {2018: 0.57, ...}
    annual_ic: dict = field(default_factory=dict)          # {2018: 0.074, ...}

    # ---- 序列(给去重 / 滚动)----
    ic_series: list | None = None          # 每日 IC(供与已入库因子算相关性)
    long_excess_nav: list | None = None    # 每日多头超额净值
    ls_nav: list | None = None             # 每日多空净值(net;给末年夏普 / 滚动 9·12 月)

    # ---- 入库判定(alphalab --gate;None=未判)----
    admission_pass: bool | None = None

    # ---- 元信息 ----
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"IC={self.ic_mean:.4f} ICIR={self.icir:.2f} t(NW)={self.t_stat_nw:.1f} | "
                f"多空 年化={self.ls_annual:.2%} 夏普={self.ls_sharpe:.2f} "
                f"Calmar={self.calmar:.2f} 最大回撤={self.ls_max_dd:.2%} | "
                f"方向={self.direction}")


class Evaluator:
    """回测器抽象。子类实现 evaluate(panel)。"""

    def evaluate(self, panel: pd.DataFrame, name: str = "factor") -> FactorMetrics:
        raise NotImplementedError

    def evaluate_expr(self, node, field_panels: dict[str, pd.DataFrame],
                      name: str = "factor") -> FactorMetrics:
        """便捷入口:在字段面板上求值表达式 → 宽表 → 回测。"""
        from engine.expression import evaluate as _eval
        panel = _eval(node, field_panels)
        m = self.evaluate(panel, name=name)
        m.expr = node.to_str()
        return m

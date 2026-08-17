# -*- coding: utf-8 -*-
"""Mock 回测器:不调 alphalab,返回合理分布的假指标,供无引擎环境下开发/测试整条流程。

分布参照真实因子量级(IC~4%、多空年化~30%、夏普~2.5 等),仅供流程跑通,数值无真实意义。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.interface import Evaluator, FactorMetrics


class MockEvaluator(Evaluator):
    def __init__(self, horizon: int = 5, seed: int | None = None,
                 years: tuple[int, int] = (2018, 2025)):
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.years = list(range(years[0], years[1] + 1))

    def evaluate(self, panel: pd.DataFrame, name: str = "factor") -> FactorMetrics:
        n = max(len(panel), 100)
        ic_mean = float(np.clip(self.rng.normal(0.04, 0.03), -0.1, 0.15))
        ic_std = float(self.rng.uniform(0.07, 0.09))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        icir_annual = icir * np.sqrt(252.0 / self.horizon)

        ls_annual = float(np.clip(self.rng.normal(0.30, 0.20), -0.2, 0.8))
        ls_sharpe = float(np.clip(self.rng.normal(2.5, 1.5), -1.0, 5.0))
        ls_max_dd = float(-self.rng.uniform(0.08, 0.20))
        calmar = ls_annual / abs(ls_max_dd) if abs(ls_max_dd) > 1e-12 else float("nan")
        long_excess_annual = ls_annual * float(self.rng.uniform(0.15, 0.30))
        long_excess_sharpe = float(np.clip(self.rng.normal(1.5, 0.8), -0.5, 3.0))
        monotonicity = float(self.rng.uniform(0.80, 0.99))

        annual_ls = {y: float(np.clip(self.rng.normal(ls_annual, 0.25), -0.5, 1.0))
                     for y in self.years}
        annual_ic = {y: float(self.rng.normal(ic_mean, 0.03)) for y in self.years}

        ic_series = self.rng.normal(ic_mean, ic_std, size=n).tolist()
        # 多头超额净值:从 1.0 起,日收益 ~ long_excess_annual/252
        daily_ret = self.rng.normal(long_excess_annual / 252.0, 0.01, size=n)
        nav = np.cumprod(1.0 + daily_ret)
        long_excess_nav = nav.tolist()
        # 多空净值(net):日收益 ~ ls_annual/252
        daily_ls = self.rng.normal(ls_annual / 252.0, 0.012, size=n)
        ls_nav = np.cumprod(1.0 + daily_ls).tolist()

        return FactorMetrics(
            expr="", direction=int(self.rng.choice([1, -1])), horizon=self.horizon,
            ic_mean=ic_mean, icir=icir, icir_annual=icir_annual,
            t_stat_nw=float(self.rng.uniform(10, 30)),
            positive_ratio=float(self.rng.uniform(0.75, 0.90)),
            ls_annual=ls_annual, ls_sharpe=ls_sharpe, ls_max_dd=ls_max_dd, calmar=calmar,
            long_excess_annual=long_excess_annual, long_excess_sharpe=long_excess_sharpe,
            monotonicity=monotonicity,
            annual_ls_return=annual_ls, annual_ic=annual_ic,
            ic_series=ic_series, long_excess_nav=long_excess_nav, ls_nav=ls_nav,
            admission_pass=None, meta={"name": name, "mock": True},
        )

# -*- coding: utf-8 -*-
"""M7 验证过滤:11 项联合过滤(消费 FactorMetrics,做入库判定)。

口径:多空(ls)为「超额」主口径(用户 2026-08-11 确认)。
阈值见 engine/config.py(研报 §9)。11 项:
  1 |IC|>0.03        2 每年多空>0(2018..2025)  3 整体多空年化>0
  4 整体夏普>0.5     5 末年夏普>0.5(从 ls_nav 末 252 日)  6 Calmar>1.0
  7 近9月多空>0      8 近12月多空>0(滚动,相对末日,从 ls_nav)
  9 IC 相关性<0.70(与已入库)  10 FSA 结构去重  11 失败模式库排除
  12 多头超额年化>0(用户加严)  13 分组单调性>0.85(用户加严)  14 ICIR>0.3(用户加严)
  15 同构子树家族上限(用户 2026-08-17,研报 §17):候选任一 ≥4 节点子树骨架
     已在 ≥2 个库存因子中出现 → 质量全面更优(×1.05)则替换该族全部旧因子(保优淘劣),
     否则拒(FSA 只对整树去重,半树相同结构会绕过;直接拒会误杀同族更优者,故带替换)

NaN 指标一律视为不通过(保守)。返回 FilterResult(passed, reasons)。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from backtest.interface import FactorMetrics
from engine.config import (
    BACKTEST_YEARS, CALMAR_MIN, FAMILY_SUBTREE_CAP, FAMILY_SUBTREE_MIN_NODES,
    IC_CORR_MAX, IC_GATE, ICIR_MIN, LONG_EXCESS_MIN,
    MONOTONICITY_MIN, ROLLING_MONTHS, SHARPE_MIN,
)
from engine.expression import Node, parse
from engine.fsa import skeleton

TD_PER_MONTH = 21     # 交易日/月(近似)
TD_PER_YEAR = 252     # 交易日/年


@dataclass
class FilterResult:
    passed: bool
    reasons: list = field(default_factory=list)
    replace_hashes: list = field(default_factory=list)   # #9 高相关更优 / #15 同族更优 → 替换这些已入库因子

    def __bool__(self) -> bool:
        return self.passed


def _quality(ic_mean: float, icir: float, monotonicity: float,
             long_excess_annual: float) -> float:
    """综合质量分(等权归一化):IC/ICIR/单调性/多头超额 各按入库阈值归一,越高越好。"""
    return (abs(ic_mean) / 0.03
            + icir / 0.30
            + monotonicity / 0.85
            + max(long_excess_annual, 0.0) / 0.05)


def _daily_returns(nav: list) -> np.ndarray:
    a = np.asarray(nav, dtype=float)
    return np.diff(a) / a[:-1]


def _ann_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(TD_PER_YEAR))


def _trailing_return(nav: list, n_days: int) -> float:
    a = np.asarray(nav, dtype=float)
    if len(a) < n_days + 1:
        return float("nan")
    return float(a[-1] / a[-n_days - 1] - 1)


def _subtree_skeletons(node: Node, min_nodes: int = FAMILY_SUBTREE_MIN_NODES) -> set[str]:
    """收集节点数 ≥ min_nodes 的全部子树骨架(含整树自身;去重)。

    例:zscore(max(min(down_shadow,120),5)) 含 ≥4 节点子树 zscore(max(min(FLD,N),N))。
    """
    out: set[str] = set()

    def _size(n: Node) -> int:
        if n.is_leaf():
            return 1
        s = 1 + sum(_size(c) for c in n.children)
        if s >= min_nodes:
            out.add(skeleton(n))
        return s

    _size(node)
    return out


def apply_filters(
    metrics: FactorMetrics,
    *,
    fsa=None,
    node=None,
    stored_ic_series: list | None = None,
    stored_factors: list | None = None,
    failed_hashes: set | None = None,
    expr_hash: str | None = None,
    years: tuple = BACKTEST_YEARS,
    rolling_months: tuple = ROLLING_MONTHS,
    ic_gate: float = IC_GATE,
    icir_min: float = ICIR_MIN,
    ic_corr_max: float = IC_CORR_MAX,
    sharpe_min: float = SHARPE_MIN,
    calmar_min: float = CALMAR_MIN,
) -> FilterResult:
    """对单个因子跑 11 项过滤。任一不过→passed=False,reasons 记全部失败项。
    高相关但综合质量全面更优 → passed=True 且 replace_hashes 列出所有被替换的旧因子。"""
    reasons: list[str] = []
    replace_hashes: list[str] = []

    # 1) |IC| > 0.03
    if not (abs(metrics.ic_mean) > ic_gate):
        reasons.append(f"1.|IC|={metrics.ic_mean:.4f}≤{ic_gate}")

    # 14) ICIR > 0.3(用户加严:IC 信息比,稳定性)
    if not (metrics.icir > icir_min):
        reasons.append(f"14.ICIR={metrics.icir:.3f}≤{icir_min}")

    # 2) 每年多空 > 0(且年份齐全)
    yr_lo, yr_hi = years
    expected = set(range(yr_lo, yr_hi + 1))
    annual = metrics.annual_ls_return
    missing = sorted(expected - set(annual))
    if missing:
        reasons.append(f"2.缺年份:{missing}")
    bad = sorted(y for y in expected if y in annual and not (annual[y] > 0))
    if bad:
        reasons.append(f"2.多空≤0的年份:{bad}")

    # 3) 整体多空年化 > 0
    if not (metrics.ls_annual > 0):
        reasons.append(f"3.整体多空年化={metrics.ls_annual:.3f}≤0")

    # 4) 整体夏普 > 0.5
    if not (metrics.ls_sharpe > sharpe_min):
        reasons.append(f"4.整体夏普={metrics.ls_sharpe:.3f}≤{sharpe_min}")

    # 5) 末年夏普 > 0.5(从 ls_nav 末 ~252 个交易日)
    if metrics.ls_nav:
        sh_last = _ann_sharpe(_daily_returns(metrics.ls_nav[-TD_PER_YEAR:]))
        if not (sh_last > sharpe_min):
            reasons.append(f"5.末年夏普={sh_last:.3f}≤{sharpe_min}")
    else:
        reasons.append("5.无 ls_nav,无法算末年夏普")

    # 6) Calmar > 1.0
    if not (metrics.calmar > calmar_min):
        reasons.append(f"6.Calmar={metrics.calmar:.3f}≤{calmar_min}")

    # 7) 近 9 月多空 > 0、8) 近 12 月多空 > 0(滚动,相对末日)
    if metrics.ls_nav:
        for m in rolling_months:
            r = _trailing_return(metrics.ls_nav, m * TD_PER_MONTH)
            if not (r > 0):
                reasons.append(f"{7 if m == rolling_months[0] else 8}.近{m}月多空={r:.3f}≤0")
    else:
        reasons.append("7/8.无 ls_nav,无法算滚动多空")

    # 12) 多头超额年化 > 0(用户加严:因子也得多头有效,不能只多空)
    if not (metrics.long_excess_annual > LONG_EXCESS_MIN):
        reasons.append(f"12.多头超额年化={metrics.long_excess_annual:.3f}≤{LONG_EXCESS_MIN}")

    # 13) 分组单调性 > 0.85(用户加严:分组收益需单调)
    if not (metrics.monotonicity > MONOTONICITY_MIN):
        reasons.append(f"13.单调性={metrics.monotonicity:.3f}≤{MONOTONICITY_MIN}")

    # 9) IC 相关性 < 0.70(与已入库;高相关但综合质量全面更优 → 替换所有相关旧因子)
    if stored_factors is not None and metrics.ic_series:
        arr = np.asarray(metrics.ic_series, dtype=float)
        new_q = _quality(metrics.ic_mean, metrics.icir, metrics.monotonicity,
                         metrics.long_excess_annual)
        correlated: list[tuple[float, dict]] = []
        for f in stored_factors:
            s = f.get("ic_series")
            if not s:
                continue
            b = np.asarray(s, dtype=float)
            n = min(len(arr), len(b))
            if n < 5:
                continue
            c = np.corrcoef(arr[-n:], b[-n:])[0, 1]
            if not np.isnan(c) and abs(c) >= ic_corr_max:
                correlated.append((abs(c), f))
        if correlated:
            # 只有综合分全面优于所有相关旧因子(各自 ×1.05)才替换全部;否则拒
            all_better = True
            for _c, f2 in correlated:
                om2 = f2.get("metrics") or {}
                old_q = _quality(om2.get("ic_mean", 0), om2.get("icir", 0),
                                 om2.get("monotonicity", 0), om2.get("long_excess_annual", 0))
                if not (new_q > old_q * 1.05):
                    all_better = False
                    break
            if all_better:
                replace_hashes = [f2.get("hash") for _c, f2 in correlated]
            else:
                best_corr = max(c for c, _ in correlated)
                reasons.append(f"9.IC相关性={best_corr:.3f}≥{ic_corr_max}")
    elif stored_ic_series and metrics.ic_series:
        # 旧版(仅 IC 序列,无质量比较)——向后兼容
        arr = np.asarray(metrics.ic_series, dtype=float)
        max_corr = 0.0
        for s in stored_ic_series:
            b = np.asarray(s, dtype=float)
            n = min(len(arr), len(b))
            if n < 5:
                continue
            c = np.corrcoef(arr[-n:], b[-n:])[0, 1]
            if not np.isnan(c) and abs(c) > max_corr:
                max_corr = abs(c)
        if max_corr >= ic_corr_max:
            reasons.append(f"9.IC相关性={max_corr:.3f}≥{ic_corr_max}")

    # 10) FSA 结构去重
    if fsa is not None and node is not None:
        ok, why = fsa.check(node)
        if not ok:
            reasons.append(f"10.{why}")

    # 11) 失败模式库排除
    if failed_hashes and expr_hash and expr_hash in failed_hashes:
        reasons.append("11.命中失败模式库")

    # 15) 同构子树家族上限(用户 2026-08-17):候选任一 ≥4 节点子树骨架已在 ≥CAP 个库存因子中
    #     出现 → 保优淘劣语义(用户 2026-08-17 升级):质量全面更优(×1.05,同 #9 门槛)则
    #     替换该族全部旧因子入库,否则拒。防「半树相同、整树不同」的家族刷量(#9 的 IC 相关门
    #     可被同结构换字段绕过,实测 max(min(·,N),N) 右半树 6/34),同时不误杀同族更优挑战者。
    if stored_factors and node is not None:
        new_subs = _subtree_skeletons(node)
        subs_per: list[set] = []
        cnt: Counter = Counter()
        for f in stored_factors:
            expr = f.get("expr")
            if not expr:
                subs_per.append(set())
                continue
            try:
                s = _subtree_skeletons(parse(expr))
            except Exception:  # noqa: BLE001  历史表达式解析失败 → 视为无子树
                s = set()
            subs_per.append(s)
            for sk in new_subs & s:       # 每个旧因子对每个骨架至多计 1 次
                cnt[sk] += 1
        over = {sk for sk, c in cnt.items() if c >= FAMILY_SUBTREE_CAP}
        if over:
            fam_members = [f for f, s in zip(stored_factors, subs_per) if s & over]
            new_q15 = _quality(metrics.ic_mean, metrics.icir, metrics.monotonicity,
                               metrics.long_excess_annual)
            all_better = True
            for f2 in fam_members:
                om2 = f2.get("metrics") or {}
                old_q = _quality(om2.get("ic_mean", 0), om2.get("icir", 0),
                                 om2.get("monotonicity", 0), om2.get("long_excess_annual", 0))
                if not (new_q15 > old_q * 1.05):
                    all_better = False
                    break
            if all_better:
                replace_hashes.extend(f2.get("hash") for f2 in fam_members)
                replace_hashes = list(dict.fromkeys(replace_hashes))
            else:
                sk0 = sorted(over)[0]
                reasons.append(f"15.同构子树家族超限({sk0} 已{len(fam_members)}个,质量未全面更优)")

    return FilterResult(passed=not reasons, reasons=reasons, replace_hashes=replace_hashes)

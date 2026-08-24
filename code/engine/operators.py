# -*- coding: utf-8 -*-
"""M1 算子库(14 种)+ 注册表。

面板约定:**宽表** —— index=DateTimeIndex(date), columns=order_book_id, values=该字段值。
  - 时序算子:组内(每只股票沿 date)滚动,窗口 n。
  - 截面算子:每个 date 截面上跨股票。
  - 逐元素算子:两个面板按 index/columns 对齐做四则。

所有时序算子 min_periods=n(前 n-1 行为 NaN,作预热期,避免部分窗口估计)。
跨量纲维度(供 review.py 过滤 #3 用)见本文件末 FIELD_DIM / OP 维度规则。

窗口范围(docs/项目执行指南.md M1):
  ma 3~250; std/max/min/rank_ts 5~120; roc/delta 3~60; skew 10~120。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# 时序算子(unary, 窗口 n)
# ============================================================================

def op_ma(p: pd.DataFrame, n: int) -> pd.DataFrame:
    return p.rolling(n, min_periods=n).mean()


def op_std(p: pd.DataFrame, n: int) -> pd.DataFrame:
    return p.rolling(n, min_periods=n).std(ddof=1)


def op_max(p: pd.DataFrame, n: int) -> pd.DataFrame:
    return p.rolling(n, min_periods=n).max()


def op_min(p: pd.DataFrame, n: int) -> pd.DataFrame:
    return p.rolling(n, min_periods=n).min()


def op_roc(p: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日变化率(收益率):p[t]/p[t-n] − 1。除零产生的 ±inf → NaN(与 op_div 同口径;
    否则 inf 会毒化截面 zscore 的均值/标准差,2019-04-18/19 数据事故曾因此放大成 40 日瘫痪)。"""
    r = p.pct_change(periods=n)
    return r.replace([np.inf, -np.inf], np.nan)


def op_delta(p: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日差分:p[t] − p[t-n]。"""
    return p.diff(periods=n)


def op_skew(p: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日滚动偏度(调整 Fisher-Pearson G1,与 Series.skew() 同口径)。

    不用 pandas rolling().skew():实测其在大量「窗口完整、方差非零」的窗口上返回 NaN
    (2026-08-17 排查:2018-2025 adj_close 面板 53 万格,rolling=NaN 而 Series.skew() 正常),
    导致 skew(·,40) 覆盖率被压到 54%。此处用滚动一/二/三阶矩和实现,语义精确等价。
    """
    s1 = p.rolling(n, min_periods=n).sum()
    s2 = (p * p).rolling(n, min_periods=n).sum()
    s3 = (p ** 3).rolling(n, min_periods=n).sum()
    m2 = (s2 - s1 * s1 / n) / n
    m3 = (s3 - 3.0 * s1 * s2 / n + 2.0 * s1 ** 3 / n ** 2) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = m3 / m2.pow(1.5)          # m2=0(常数窗)→ 0/0=NaN,与 pandas 行为一致
    adj = float(np.sqrt(n * (n - 1)) / (n - 2))   # G1 调整系数(窗口范围 ≥10,n>2 恒成立)
    return g1 * adj


def _ts_rank_last(w: np.ndarray) -> float:
    """窗口内最后一个值的升序排名(0-based)归一到 [0,1]:(rank)/(n-1)。"""
    n = len(w)
    if n <= 1:
        return np.nan
    order = w.argsort(kind="stable")  # 升序位置
    rank0 = int(np.where(order == n - 1)[0][0])  # 末元素在排序后的位次
    return rank0 / (n - 1)


def op_rank_ts(p: pd.DataFrame, n: int) -> pd.DataFrame:
    """时序排名:当前值在过去 n 日的升序百分位 ∈ [0,1]。"""
    return p.rolling(n, min_periods=n).apply(_ts_rank_last, raw=True)


# ============================================================================
# 逐元素算子(binary)
# ============================================================================

def op_add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a + b


def op_sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a - b


def op_mul(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a * b


def op_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    r = a / b
    return r.replace([np.inf, -np.inf], np.nan)  # 除零 → NaN(避免毒化截面)


# ============================================================================
# 截面算子(unary)
# ============================================================================

def op_zscore(p: pd.DataFrame) -> pd.DataFrame:
    """逐截面标准化:(x − 截面均值)/ 截面标准差。入口先消毒 ±inf → NaN:
    单个 inf 就会毒化整个截面的 mean/std,令当天全截面 NaN(2019-04-18 事故的放大器),
    必须挡在这里——叶子与 div/roc 也已消毒,此处为最后防线。"""
    p = p.replace([np.inf, -np.inf], np.nan)
    mu = p.mean(axis=1)
    sd = p.std(axis=1, ddof=1)
    return p.sub(mu, axis=0).div(sd, axis=0)


def op_rank_cs(p: pd.DataFrame) -> pd.DataFrame:
    """逐截面排名,归一百分位 ∈ [0,1]。"""
    return p.rank(axis=1, pct=True)


# ============================================================================
# 注册表:op → (kind, arity, window_range|None, func, dim_out)
#   dim_out:输出维度如何由输入推导
#     "same"    = 保留操作数维度(时序 ma/std/max/min、delta、逐元素 add/sub)
#     "dimless" = 无量纲(roc/skew/rank_ts、截面 zscore/rank_cs)
#     "ratio"   = 无量纲比(逐元素 mul/div)
# ============================================================================

TS_OPS: dict[str, tuple] = {
    "ma":      (op_ma,      (3, 250),  "same"),
    "std":     (op_std,     (5, 120),  "same"),
    "max":     (op_max,     (5, 120),  "same"),
    "min":     (op_min,     (5, 120),  "same"),
    "roc":     (op_roc,     (3, 60),   "dimless"),
    "delta":   (op_delta,   (3, 60),   "same"),
    "skew":    (op_skew,    (10, 120), "dimless"),
    "rank_ts": (op_rank_ts, (5, 120),  "dimless"),
}
ELEM_OPS: dict[str, tuple] = {
    "add": (op_add, "same"),
    "sub": (op_sub, "same"),
    "mul": (op_mul, "ratio"),
    "div": (op_div, "ratio"),
}
CS_OPS: dict[str, tuple] = {
    "zscore":  (op_zscore,  "dimless"),
    "rank_cs": (op_rank_cs, "dimless"),
}

OP_REGISTRY: dict[str, dict] = {}
for _name, (_f, _wr, _dim) in TS_OPS.items():
    OP_REGISTRY[_name] = {"kind": "ts", "arity": 1, "window_range": _wr,
                          "func": _f, "dim": _dim}
for _name, (_f, _dim) in ELEM_OPS.items():
    OP_REGISTRY[_name] = {"kind": "elem", "arity": 2, "window_range": None,
                          "func": _f, "dim": _dim}
for _name, (_f, _dim) in CS_OPS.items():
    OP_REGISTRY[_name] = {"kind": "cs", "arity": 1, "window_range": None,
                          "func": _f, "dim": _dim}

NUM_OPS = len(OP_REGISTRY)  # = 14
TS_OP_NAMES = list(TS_OPS.keys())
ELEM_OP_NAMES = list(ELEM_OPS.keys())
CS_OP_NAMES = list(CS_OPS.keys())
ALL_OP_NAMES = list(OP_REGISTRY.keys())


def apply(name: str, args: list[pd.DataFrame], window: int | None = None) -> pd.DataFrame:
    """按算子名分发求值。args 为操作数面板列表;时序算子需 window。"""
    if name not in OP_REGISTRY:
        raise KeyError(f"未知算子: {name}")
    info = OP_REGISTRY[name]
    func = info["func"]
    if info["kind"] == "ts":
        if window is None:
            raise ValueError(f"时序算子 {name} 需 window")
        return func(args[0], window)
    if info["kind"] == "cs":
        return func(args[0])
    return func(args[0], args[1])  # elem


# ============================================================================
# 字段维度(M2 字段 → 量纲组),供 review.py 跨量纲过滤用
#   规则:add/sub 要求两操作数同量纲(或同属 dimless);mul/div 恒输出无量纲。
# ============================================================================

DIM_PRICE = "price"        # 原始/前复权价、prev_close
DIM_MV = "mv"              # 市值(price×shares,与 price 不同量级)
DIM_VOLUME = "volume"
DIM_AMOUNT = "amount"
DIM_SHARES = "shares"
DIM_DIMLESS = "dimless"    # 收益/比率(可相互组合):ret/pct/overnight/intraday/amplitude/shadows/hl_ratio
DIM_LOGVOL = "log_volume"
DIM_LOGAMT = "log_amount"
DIM_LOGMV = "log_mv"

FIELD_DIM: dict[str, str] = {
    "open": DIM_PRICE, "high": DIM_PRICE, "low": DIM_PRICE, "close": DIM_PRICE,
    "prev_close": DIM_PRICE,
    "adj_open": DIM_PRICE, "adj_high": DIM_PRICE, "adj_low": DIM_PRICE, "adj_close": DIM_PRICE,
    "prev_adj_close": DIM_PRICE,
    "volume": DIM_VOLUME, "amount": DIM_AMOUNT,
    "mv": DIM_MV,
    "free_circulation": DIM_SHARES,
    "pct": DIM_DIMLESS, "ret": DIM_DIMLESS,
    "overnight": DIM_DIMLESS, "intraday": DIM_DIMLESS,
    "amplitude": DIM_DIMLESS, "up_shadow": DIM_DIMLESS, "down_shadow": DIM_DIMLESS,
    "hl_ratio": DIM_DIMLESS,
    "log_volume": DIM_LOGVOL, "log_amount": DIM_LOGAMT, "log_mv": DIM_LOGMV,
    # 基本面 6 字段(2026-08-24):全为比率/收益率型 → dimless,可与 rank 类组合;
    # 财务指标为公告时点阶梯(季更),时序算子语义如 roc(roe,60)=盈利能力改善动量
    "roe": DIM_DIMLESS, "roa": DIM_DIMLESS, "profit_growth": DIM_DIMLESS,
    "bm": DIM_DIMLESS, "div_yield": DIM_DIMLESS, "ps": DIM_DIMLESS,
}


def field_dimension(field: str) -> str:
    """取叶子字段的量纲组;未知字段默认 dimless(保守允许组合)。"""
    return FIELD_DIM.get(field, DIM_DIMLESS)

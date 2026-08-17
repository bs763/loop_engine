# -*- coding: utf-8 -*-
"""派生字段计算(严格按 docs/项目执行指南.md M2 §2.3 公式,全部基于前复权价)。

输入:data_loader.build_factor_table() 产出的 base 表
      (含 adj_open/high/low/close, volume, amount, mv, free_circulation, order_book_id, date)。
输出:追加 10 个派生字段 + pct + prev_adj_close。

阶段 1 已确认的口径要点:
  - overnight 的"昨收"用 shift(adj_close,1),**不用** daily_bar.prev_close
    (prev_close 在除权日是除权基准价、平时是原始价,口径不一致);
  - 所有比率为前复权价的比率;volume / amount / mv 不复权,取对数;
  - prev_adj_close 为组内(adj_close 按 date)的前一个交易日值(daily_bar 已省略停牌日,
    故停牌复牌的收益自然跨越停牌期,符合常规)。

恒等式(供验收): (1+overnight) × (1+intraday) − 1 == ret;  pct/100 == ret。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# M2 的 10 个派生字段(规范名)
DERIVED_FIELDS: list[str] = [
    "intraday", "overnight", "amplitude", "up_shadow", "down_shadow", "hl_ratio",
    "ret", "log_volume", "log_amount", "log_mv",
]
# 辅助列(非 M2 的 10 派生,但下游/验收需要)
AUX_FIELDS: list[str] = ["prev_adj_close", "pct"]


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """在 base 表上追加派生字段。

    会按 (order_book_id, date) 排序后计算(组内 shift 依赖此顺序)。返回新 DataFrame。
    """
    required = {"order_book_id", "date", "adj_open", "adj_high", "adj_low", "adj_close",
                "volume", "amount", "mv"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_derived 缺少必需列: {sorted(missing)}")

    df = df.sort_values(["order_book_id", "date"]).reset_index(drop=True)

    # 昨收 = 组内前复权收盘的前一个交易日(shift 1)
    prev = df.groupby("order_book_id", sort=False)["adj_close"].shift(1)
    df["prev_adj_close"] = prev

    # 收益类
    df["pct"] = (df["adj_close"] / prev - 1.0) * 100.0   # 涨跌幅%(M2 自算,用前复权 close)
    df["ret"] = df["pct"] / 100.0                         # 收益率小数
    df["intraday"] = df["adj_close"] / df["adj_open"] - 1.0
    df["overnight"] = df["adj_open"] / prev - 1.0          # 隔夜(昨收=前复权 shift1)

    # K 线结构类(分母统一用 adj_close)
    df["amplitude"] = (df["adj_high"] - df["adj_low"]) / df["adj_close"]
    body_hi = np.maximum(df["adj_open"], df["adj_close"])
    body_lo = np.minimum(df["adj_open"], df["adj_close"])
    df["up_shadow"] = (df["adj_high"] - body_hi) / df["adj_close"]
    df["down_shadow"] = (body_lo - df["adj_low"]) / df["adj_close"]
    hl_range = df["adj_high"] - df["adj_low"]
    df["hl_ratio"] = (df["adj_close"] - df["adj_low"]) / hl_range  # 一字板(high==low)分母0 → NaN

    # 规模类(不复权量/额/市值,取对数)
    df["log_volume"] = np.log(df["volume"] + 1.0)
    df["log_amount"] = np.log(df["amount"] + 1.0)
    df["log_mv"] = np.log(df["mv"] + 1.0)

    # inf → NaN(一字板 hl_ratio、零成交等边界),交由截面标准化处理
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

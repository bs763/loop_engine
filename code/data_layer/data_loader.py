# -*- coding: utf-8 -*-
"""数据加载层:OSS 取数 + 本地缓存 + 前复权 + 股本/标记对齐。

职责(严格按 docs/项目执行指南.md M2 口径,阶段 1 已确认):
  1. 从 OSS 拉取(或读本地缓存)原始 daily_bar / ex_factor / shares / is_st / is_suspended;
  2. 复权:adj_price = raw × cum_at_t(**纯时点累计复权,无未来锚**)
       - cum_at_t:ex_cum_factor 按 ex_date **ASOF 前向填充**(每个交易日取 ex_date≤当日 的最近一次);
         早于首次除权事件的日期无因子 → 视为 1.0(尚未发生任何除权);
       - 2026-08-18 未来函数审计:旧口径 `raw × f_t / cum_latest` 的 cum_latest 取自数据末尾
         (含未来公司行动),比率类用法虽精确相消,但 delta/价差/水平类运算把逐股 1/L 的
         未来信息带入截面 → 移除 /cum_latest(10 个派生字段数学上逐值不变,见 derived_fields.py);
  3. 对齐 shares.free_circulation → mv = close × free_circulation(**时点自由流通市值,不复权**;
     复权价只用于 adj_* 与比率字段——水平字段用前复权价会把"全历史最新除权基准"(含未来
     公司行动)引入截面排名,2026-08-18 未来函数审计修正,见 mv口径修正方案.md);
  4. 对齐 is_st / is_suspended 标记列(**字段层不过滤**,留给 M7 回测过滤);
  5. 输出 base 表(adj OHLC + volume + amount + mv + free_circulation + 标记)。

派生字段(overnight/intraday/amplitude/... 比率与对数)在 derived_fields.py 计算。

缓存策略:只缓存「昂贵的 OSS 原始拉取」(按年落 parquet 到 cache/);
前复权 / 对齐 / 派生每次现算(DuckDB + pandas 内存计算,快,且复权锚定始终取最新)。
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from data_layer import oss
from paths import CACHE_DIR

# ---- 本地缓存子目录 ----
_C_DAILY = CACHE_DIR / "daily_bar"
_C_SHARES = CACHE_DIR / "shares"
_C_ST = CACHE_DIR / "is_st"
_C_SUSP = CACHE_DIR / "is_suspended"
_C_EXF = CACHE_DIR / "ex_factor.parquet"
_C_FININD = CACHE_DIR / "fin_indicators"
_C_VALU = CACHE_DIR / "valuation"

# (kind → (OSS glob, 本地目录))
_TABLE_MAP = {
    "daily_bar": (oss.DAILY_BAR, _C_DAILY),
    "shares": (oss.SHARES, _C_SHARES),
    "is_st": (oss.IS_ST, _C_ST),
    "is_suspended": (oss.IS_SUSPENDED, _C_SUSP),
    "fin_indicators": (oss.FIN_INDICATORS, _C_FININD),
    "valuation": (oss.VALUATION, _C_VALU),
}

# 基本面字段选取与改名(2026-08-24 首批 6 个,全为比率型=dimless):
#   故意不取 PE/PEG——亏损股 PE 为负,负值在截面 rank 中语义反转(巨亏排成"最便宜"),
#   BM/PS/股息率分母恒正干净;盈利收益率留待二期从 income 表按 PE>0 条件自算。
FUNDAMENTAL_COLS = {
    "return_on_equity_ttm": "roe",
    "return_on_asset_ttm": "roa",
    "net_profit_parent_company_growth_ratio_ttm": "profit_growth",
    "book_to_market_ratio_lf": "bm",
    "dividend_yield_ttm": "div_yield",
    "ps_ratio_ttm": "ps",
}


def _sql_list(paths: list[Path]) -> str:
    """把本地 parquet 路径列表格式化成 DuckDB 列表字面量:['a','b',...]。"""
    return "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"


def _ensure_year_cache(con: duckdb.DuckDBPyConnection, kind: str, year: int,
                       *, use_cache: bool = True) -> Path:
    """确保某年某表的本地缓存 parquet 存在,返回其路径;不存在(或 use_cache=False)则从 OSS 拉。"""
    src_glob, dst_dir = _TABLE_MAP[kind]
    dst = dst_dir / f"year_{year}.parquet"
    if dst.exists() and use_cache:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{src_glob}', hive_partitioning=1) WHERE year={year}) "
        f"TO '{dst.as_posix()}' (FORMAT PARQUET)"
    )
    return dst


def _ensure_ex_factor_cache(con: duckdb.DuckDBPyConnection, *, use_cache: bool = True) -> Path:
    """复权因子(单文件、全历史、很小)→ 缓存整张表一次。"""
    if _C_EXF.exists() and use_cache:
        return _C_EXF
    _C_EXF.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{oss.EX_FACTOR}')) "
        f"TO '{_C_EXF.as_posix()}' (FORMAT PARQUET)"
    )
    return _C_EXF


def build_factor_table(start_year: int, end_year: int, *, use_cache: bool = True,
                       con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """加载 + 前复权 + 对齐 → base 表(尚未算派生字段)。

    返回列:order_book_id, date, open/high/low/close(原始), prev_close, volume, amount,
            free_circulation, cum_at_t, adj_open/high/low/close, mv, is_st, is_suspended,
            + 基本面 6 字段(roe/roa/profit_growth/bm/div_yield/ps,2026-08-24,见 FUNDAMENTAL_COLS)。
    """
    if end_year < start_year:
        raise ValueError(f"end_year({end_year}) < start_year({start_year})")
    years = list(range(start_year, end_year + 1))
    own_con = con is None
    if own_con:
        con = oss.connect()
    try:
        db_files = [_ensure_year_cache(con, "daily_bar", y, use_cache=use_cache) for y in years]
        sh_files = [_ensure_year_cache(con, "shares", y, use_cache=use_cache) for y in years]
        st_files = [_ensure_year_cache(con, "is_st", y, use_cache=use_cache) for y in years]
        su_files = [_ensure_year_cache(con, "is_suspended", y, use_cache=use_cache) for y in years]
        fi_files = [_ensure_year_cache(con, "fin_indicators", y, use_cache=use_cache) for y in years]
        va_files = [_ensure_year_cache(con, "valuation", y, use_cache=use_cache) for y in years]
        exf_file = _ensure_ex_factor_cache(con, use_cache=use_cache)

        # 基本段列清单(原名 AS 改名)拼进 SELECT
        fi_map = {k: v for k, v in FUNDAMENTAL_COLS.items()
                  if k in ("return_on_equity_ttm", "return_on_asset_ttm",
                           "net_profit_parent_company_growth_ratio_ttm")}
        va_map = {k: v for k, v in FUNDAMENTAL_COLS.items()
                  if k in ("book_to_market_ratio_lf", "dividend_yield_ttm", "ps_ratio_ttm")}
        fi_sel = ", ".join(f"fi.{k} AS {v}" for k, v in fi_map.items())
        va_sel = ", ".join(f"va.{k} AS {v}" for k, v in va_map.items())

        query = f"""
        WITH db AS (
          SELECT order_book_id, date, open, high, low, close, prev_close, volume, total_turnover
          FROM read_parquet({_sql_list(db_files)})
        ),
        exf AS (
          SELECT order_book_id, ex_date, ex_cum_factor
          FROM read_parquet('{exf_file.as_posix()}')
        ),
        sh AS (
          SELECT order_book_id, date, free_circulation
          FROM read_parquet({_sql_list(sh_files)})
        ),
        st AS (
          SELECT order_book_id, date, is_st
          FROM read_parquet({_sql_list(st_files)})
        ),
        su AS (
          SELECT order_book_id, date, is_suspended
          FROM read_parquet({_sql_list(su_files)})
        ),
        fi AS (
          SELECT order_book_id, date, {", ".join(fi_map)}
          FROM read_parquet({_sql_list(fi_files)})
        ),
        va AS (
          SELECT order_book_id, date, {", ".join(va_map)}
          FROM read_parquet({_sql_list(va_files)})
        )
        SELECT
          db.order_book_id, db.date,
          db.open, db.high, db.low, db.close, db.prev_close,
          db.volume, db.total_turnover AS amount,
          sh.free_circulation,
          COALESCE(exf.ex_cum_factor, 1.0) AS cum_at_t,
          db.open  * COALESCE(exf.ex_cum_factor, 1.0) AS adj_open,
          db.high  * COALESCE(exf.ex_cum_factor, 1.0) AS adj_high,
          db.low   * COALESCE(exf.ex_cum_factor, 1.0) AS adj_low,
          db.close * COALESCE(exf.ex_cum_factor, 1.0) AS adj_close,
          db.close * sh.free_circulation             AS mv,
          COALESCE(st.is_st, false)       AS is_st,
          COALESCE(su.is_suspended, false) AS is_suspended,
          {fi_sel},
          {va_sel}
        FROM db
        ASOF LEFT JOIN exf
          ON db.order_book_id = exf.order_book_id AND db.date >= exf.ex_date
        LEFT JOIN sh ON db.order_book_id = sh.order_book_id AND db.date = sh.date
        LEFT JOIN st ON db.order_book_id = st.order_book_id AND db.date = st.date
        LEFT JOIN su ON db.order_book_id = su.order_book_id AND db.date = su.date
        LEFT JOIN fi ON db.order_book_id = fi.order_book_id AND db.date = fi.date
        LEFT JOIN va ON db.order_book_id = va.order_book_id AND db.date = va.date
        """
        df = con.execute(query).fetchdf()
    finally:
        if own_con:
            con.close()
    return df


def load_factor_data(start_year: int, end_year: int, *, use_cache: bool = True,
                     con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """便捷入口:build_factor_table → compute_derived,返回含全部原始+派生字段的最终表。"""
    from data_layer.derived_fields import compute_derived  # 延迟导入,避免循环
    df = build_factor_table(start_year, end_year, use_cache=use_cache, con=con)
    return compute_derived(df)

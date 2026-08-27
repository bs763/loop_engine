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
_C_INCOME = CACHE_DIR / "income"
_C_BALANCE = CACHE_DIR / "balance_sheet"
_C_CASHFLOW = CACHE_DIR / "cash_flow"

# (kind → (OSS glob, 本地目录))
_TABLE_MAP = {
    "daily_bar": (oss.DAILY_BAR, _C_DAILY),
    "shares": (oss.SHARES, _C_SHARES),
    "is_st": (oss.IS_ST, _C_ST),
    "is_suspended": (oss.IS_SUSPENDED, _C_SUSP),
    "fin_indicators": (oss.FIN_INDICATORS, _C_FININD),
    "valuation": (oss.VALUATION, _C_VALU),
    "income": (oss.INCOME, _C_INCOME),
    "balance_sheet": (oss.BALANCE_SHEET, _C_BALANCE),
    "cash_flow": (oss.CASH_FLOW, _C_CASHFLOW),
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

# 基本面二期派生字段(2026-08-27 用户拍板):三表跨表比率,回测端无量纲,
# 经济逻辑=杜邦分解族(利润率×周转)与现金流质量族。全部日频 PIT 阶梯
# (实测 op_margin 每股每年 4 个 distinct 值)。分母护栏:营收/总资产 >0 且有限。
FUNDAMENTAL2_COLS = ["op_margin", "asset_turn", "ocf_asset", "ocf_margin", "debt_ratio", "np_margin"]


def _sql_list(paths: list[Path]) -> str:
    """把本地 parquet 路径列表格式化成 DuckDB 列表字面量:['a','b',...]。"""
    return "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"


def _ensure_year_cache(con: duckdb.DuckDBPyConnection, kind: str, year: int,
                       *, use_cache: bool = True, cols: list[str] | None = None) -> Path:
    """确保某年某表的本地缓存 parquet 存在,返回其路径;不存在(或 use_cache=False)则从 OSS 拉。

    cols:列裁剪(2026-08-24)——valuation 全表 1.5GB 而我们只用 3 列,拉取时只落所需列
    (含 order_book_id/date);宽表全列缓存的教训:磁盘 40x 冗余。
    """
    src_glob, dst_dir = _TABLE_MAP[kind]
    dst = dst_dir / f"year_{year}.parquet"
    if dst.exists() and use_cache:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    sel = ", ".join(cols) if cols else "*"
    con.execute(
        f"COPY (SELECT {sel} FROM read_parquet('{src_glob}', hive_partitioning=1) WHERE year={year}) "
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
        # 基本面两表列裁剪缓存(只落 join 键 + 所需列)
        fi_cols = ["order_book_id", "date"] + [k for k in FUNDAMENTAL_COLS
                                               if k in ("return_on_equity_ttm", "return_on_asset_ttm",
                                                        "net_profit_parent_company_growth_ratio_ttm")]
        va_cols = ["order_book_id", "date"] + [k for k in FUNDAMENTAL_COLS
                                               if k in ("book_to_market_ratio_lf", "dividend_yield_ttm",
                                                        "ps_ratio_ttm")]
        fi_files = [_ensure_year_cache(con, "fin_indicators", y, use_cache=use_cache, cols=fi_cols)
                    for y in years]
        va_files = [_ensure_year_cache(con, "valuation", y, use_cache=use_cache, cols=va_cols)
                    for y in years]
        ic_cols = ["order_book_id", "date", "operating_revenue_ttm_0",
                   "profit_from_operation_ttm_0", "net_profit_ttm_0"]
        bs_cols = ["order_book_id", "date", "total_assets_mrq_0", "total_liabilities_mrq_0"]
        cf_cols = ["order_book_id", "date", "cash_flow_from_operating_activities_ttm_0"]
        ic_files = [_ensure_year_cache(con, "income", y, use_cache=use_cache, cols=ic_cols)
                    for y in years]
        bs_files = [_ensure_year_cache(con, "balance_sheet", y, use_cache=use_cache, cols=bs_cols)
                    for y in years]
        cf_files = [_ensure_year_cache(con, "cash_flow", y, use_cache=use_cache, cols=cf_cols)
                    for y in years]
        exf_file = _ensure_ex_factor_cache(con, use_cache=use_cache)

        # 基本面二期派生比率(2026-08-27):三表跨表 SELECT,分母>0 且分子分母有限;
        # np_margin 允许负值(亏损=差,语义正确)
        f2 = """
          CASE WHEN isfinite(ic.profit_from_operation_ttm_0) AND ic.operating_revenue_ttm_0 > 0
               THEN ic.profit_from_operation_ttm_0 / ic.operating_revenue_ttm_0 END AS op_margin,
          CASE WHEN isfinite(ic.operating_revenue_ttm_0) AND bs.total_assets_mrq_0 > 0
               THEN ic.operating_revenue_ttm_0 / bs.total_assets_mrq_0 END AS asset_turn,
          CASE WHEN isfinite(cf.cash_flow_from_operating_activities_ttm_0) AND bs.total_assets_mrq_0 > 0
               THEN cf.cash_flow_from_operating_activities_ttm_0 / bs.total_assets_mrq_0 END AS ocf_asset,
          CASE WHEN isfinite(cf.cash_flow_from_operating_activities_ttm_0) AND ic.operating_revenue_ttm_0 > 0
               THEN cf.cash_flow_from_operating_activities_ttm_0 / ic.operating_revenue_ttm_0 END AS ocf_margin,
          CASE WHEN isfinite(bs.total_liabilities_mrq_0) AND bs.total_assets_mrq_0 > 0
               THEN bs.total_liabilities_mrq_0 / bs.total_assets_mrq_0 END AS debt_ratio,
          CASE WHEN isfinite(ic.net_profit_ttm_0) AND ic.operating_revenue_ttm_0 > 0
               THEN ic.net_profit_ttm_0 / ic.operating_revenue_ttm_0 END AS np_margin"""

        # 基本段列清单(原名 AS 改名)拼进 SELECT。
        # 护栏(2026-08-24 实测发现):ps 存在 inf(营收近零的壳公司)且负营收的负 PS 与负 PE
        # 同一语义反转陷阱 → ps 仅保留 >0 且有限;其余只挡 inf/NaN(roe/roa/growth 的负值
        # 语义正确——亏损就是差,保留)。
        fi_map = {k: v for k, v in FUNDAMENTAL_COLS.items()
                  if k in ("return_on_equity_ttm", "return_on_asset_ttm",
                           "net_profit_parent_company_growth_ratio_ttm")}
        va_map = {k: v for k, v in FUNDAMENTAL_COLS.items()
                  if k in ("book_to_market_ratio_lf", "dividend_yield_ttm", "ps_ratio_ttm")}
        fi_sel = ", ".join(
            f"CASE WHEN isfinite(fi.{k}) THEN fi.{k} END AS {v}" for k, v in fi_map.items())
        va_sel = ", ".join(
            (f"CASE WHEN isfinite(va.{k}) THEN va.{k} END AS {v}" if v != "ps"
             else f"CASE WHEN isfinite(va.{k}) AND va.{k} > 0 THEN va.{k} END AS ps")
            for k, v in va_map.items())

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
        ),
        ic AS (
          SELECT order_book_id, date, operating_revenue_ttm_0, profit_from_operation_ttm_0, net_profit_ttm_0
          FROM read_parquet({_sql_list(ic_files)})
        ),
        bs AS (
          SELECT order_book_id, date, total_assets_mrq_0, total_liabilities_mrq_0
          FROM read_parquet({_sql_list(bs_files)})
        ),
        cf AS (
          SELECT order_book_id, date, cash_flow_from_operating_activities_ttm_0
          FROM read_parquet({_sql_list(cf_files)})
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
          {va_sel},
          {f2}
        FROM db
        ASOF LEFT JOIN exf
          ON db.order_book_id = exf.order_book_id AND db.date >= exf.ex_date
        LEFT JOIN sh ON db.order_book_id = sh.order_book_id AND db.date = sh.date
        LEFT JOIN st ON db.order_book_id = st.order_book_id AND db.date = st.date
        LEFT JOIN su ON db.order_book_id = su.order_book_id AND db.date = su.date
        LEFT JOIN fi ON db.order_book_id = fi.order_book_id AND db.date = fi.date
        LEFT JOIN va ON db.order_book_id = va.order_book_id AND db.date = va.date
        LEFT JOIN ic ON db.order_book_id = ic.order_book_id AND db.date = ic.date
        LEFT JOIN bs ON db.order_book_id = bs.order_book_id AND db.date = bs.date
        LEFT JOIN cf ON db.order_book_id = cf.order_book_id AND db.date = cf.date
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

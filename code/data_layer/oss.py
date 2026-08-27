# -*- coding: utf-8 -*-
"""OSS 内网直读模块(DuckDB + 阿里云 ECS RAM 角色 STS 凭证,无需 AccessKey)。

依据《factor_loop_engine/OSS取数说明.md》(2026-08 更新版):
  - bucket 已从 `wengxinhua` 迁移到 **`ys-datalake`**;
  - 路径前缀由 `quantdb/stock/...` 改为 **`stock/...`**(去掉 `quantdb/`);
  - 鉴权走实例元数据服务(IMDSv2)取 STS 临时凭证,DuckDB httpfs 用 S3 兼容协议
    走 OSS 内网端点(免公网流量费、低延迟);临时凭证数小时过期,每次 connect() 重取。

注意:根目录的 duckdb_daily_bar.py、alpha_framework/oss.py、factor_work/oss.py 三份
旧代码仍指向旧 bucket `wengxinhua`,已失效——本模块是项目内**唯一**生效的取数入口,
后续所有取数统一走这里。
"""
from __future__ import annotations

import json
import urllib.request

import duckdb

# 阿里云实例元数据服务(IMDSv2)
META = "http://100.100.100.200/latest"
# OSS 内网端点(免公网流量费、低延迟)
INTERNAL_ENDPOINT = "oss-cn-shanghai-internal.aliyuncs.com"
REGION = "cn-shanghai"

# ============================================================================
# 数据集路径(bucket = ys-datalake,2026-08 迁移后口径)
# ============================================================================
BUCKET = "ys-datalake"

# ---- 行情 ----
DAILY_BAR = f"s3://{BUCKET}/stock/daily_bar/year=*/data.parquet"        # A股日频OHLCV(未复权原始价)
MINUTE_BAR = f"s3://{BUCKET}/stock/minute_bar/date=*/data.parquet"      # A股分钟K线
OPEN_AUCTION = f"s3://{BUCKET}/stock/open_auction/year=*/data.parquet"  # 开盘集合竞价

# ---- 复权 / 股本 / 标记(因子计算依赖)----
EX_FACTOR = f"s3://{BUCKET}/stock/ex_factor/data.parquet"               # 复权因子(单文件,无分区);含 ex_cum_factor
SHARES = f"s3://{BUCKET}/stock/shares/year=*/data.parquet"             # 每日股本快照;含 free_circulation(自由流通股本)
IS_ST = f"s3://{BUCKET}/stock/is_st/year=*/data.parquet"              # ST 标记(BOOLEAN)
IS_SUSPENDED = f"s3://{BUCKET}/stock/is_suspended/year=*/data.parquet"  # 停牌标记(BOOLEAN)

# ---- 基本面(2026-08-24 接入;文档速查表漏记,已实探验证)----
# 两表均为日频、按公告时点 PIT 对齐(实测:ROE 每股每年仅 ~5 个 distinct 值,
# 跳变精确集中在 4/8/10 月披露季,3 月为快报窗口——阶梯干净无未来函数嫌疑)
FIN_INDICATORS = f"s3://{BUCKET}/stock/fin_indicators/year=*/data.parquet"  # ROE/ROA/净利增速
VALUATION = f"s3://{BUCKET}/stock/valuation/year=*/data.parquet"            # BM/PS/PE/股息率/市值等

# ---- 基本面二期三表(2026-08-27 用户拍板扩字段;同为日频 PIT 阶梯,实测 op_margin 每股每年 4 个 distinct 值)----
INCOME = f"s3://{BUCKET}/stock/income/year=*/data.parquet"                  # 利润表(mrq/ttm 全科目)
BALANCE_SHEET = f"s3://{BUCKET}/stock/balance_sheet/year=*/data.parquet"    # 资产负债表
CASH_FLOW = f"s3://{BUCKET}/stock/cash_flow/year=*/data.parquet"            # 现金流量表(OCF)

# ---- 标的 / 日历 / 指数(后续过滤/分池用)----
INSTRUMENTS = f"s3://{BUCKET}/stock/instruments/data.parquet"          # 标的清单
FACTOR_EXPO_V2 = f"s3://{BUCKET}/stock/factor_exposure_v2/year=*/data.parquet"  # Barra CNE5 因子暴露
TRADING_DATES = f"s3://{BUCKET}/common/trading_dates/data.parquet"     # 交易日历
YIELD_CURVE = f"s3://{BUCKET}/common/yield_curve/year=*/data.parquet"  # 国债收益率曲线(无风险利率)
INDEX_COMP = f"s3://{BUCKET}/index/components/year=*/data.parquet"     # 指数成分股
INDEX_WEIGHTS = f"s3://{BUCKET}/index/weights/year=*/data.parquet"     # 指数权重


def _imds(path: str, token: str | None = None, method: str = "GET",
          extra: dict | None = None, timeout: int = 15) -> str:
    """访问实例元数据服务(IMDSv2,需带 token)。"""
    headers: dict[str, str] = {}
    if token:
        headers["X-aliyun-ecs-metadata-token"] = token
    if extra:
        headers.update(extra)
    req = urllib.request.Request(META + path, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def get_token(ttl: int = 1800) -> str:
    """申请 IMDSv2 令牌(默认有效期 1800 秒)。"""
    return _imds(
        "/api/token",
        method="PUT",
        extra={"X-aliyun-ecs-metadata-token-ttl-seconds": str(ttl)},
    )


def get_creds() -> dict:
    """取本实例 RAM 角色的 STS 临时凭证。

    返回 dict,含 AccessKeyId / AccessKeySecret / SecurityToken / Expiration。
    凭证数小时过期,每次调用都重新获取,不缓存。
    """
    tok = get_token()
    # 先查角色名,再用 token 取该角色的临时凭证
    role = _imds("/meta-data/ram/security-credentials/", token=tok).strip()
    raw = _imds(f"/meta-data/ram/security-credentials/{role}", token=tok)
    return json.loads(raw)


def connect(region: str = REGION, endpoint: str = INTERNAL_ENDPOINT,
            use_ssl: bool = True) -> duckdb.DuckDBPyConnection:
    """建立一个已配置好 S3 兼容(OSS)访问的 DuckDB 内存连接。

    每次调用都重新取临时凭证(凭证会过期,不缓存)。
    返回的连接可直接对 s3:// 路径做 read_parquet 查询。
    """
    creds = get_creds()
    con = duckdb.connect()
    # 加载 httpfs 扩展(S3/OSS 读取所需);优先 LOAD,失败则先 INSTALL 再 LOAD
    try:
        con.execute("LOAD httpfs;")
    except Exception:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    # 配置 OSS(S3 兼容)访问参数
    con.execute(f"SET s3_region='{region}'")
    con.execute(f"SET s3_endpoint='{endpoint}'")
    con.execute(f"SET s3_use_ssl={'true' if use_ssl else 'false'}")
    con.execute("SET s3_url_style='vhost'")
    con.execute(f"SET s3_access_key_id='{creds['AccessKeyId']}'")
    con.execute(f"SET s3_secret_access_key='{creds['AccessKeySecret']}'")
    con.execute(f"SET s3_session_token='{creds['SecurityToken']}'")
    return con


def read_parquet(con: duckdb.DuckDBPyConnection, path: str, *,
                 hive_partitioning: bool = True) -> "duckdb.DuckDBPyRelation":
    """便捷封装:对某个 OSS 路径返回 DuckDB 关系(可链式 .filter/.select)。

    默认开启 hive 分区裁剪(year=/date= 等分区列自动识别),配合 WHERE 用分区列裁剪可大幅少读数据。
    """
    opts = "hive_partitioning=true" if hive_partitioning else "hive_partitioning=false"
    return con.from_query(f"SELECT * FROM read_parquet('{path}', {opts})")

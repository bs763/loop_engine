# 阿里云 ECS 从 OSS 取量化数据（RAM 角色 + DuckDB，无需 AccessKey）

## 适用场景
在这台阿里云 ECS（cn-shanghai）上，用 DuckDB 直读 OSS 里的 parquet 行情/因子数据，
做因子计算、回测、分析。无需配置任何 AK/SK，靠实例绑定的 RAM 角色自动鉴权。

> ⚠️ **2026-08 更新**：原 bucket `wengxinhua` 已删除，数据迁移到 **`ys-datalake`**。
> 路径前缀由原来的 `quantdb/stock/...` 改为 **`stock/...`**（去掉 `quantdb/`），
> `quantdb/common/...` 改为 **`common/...`**。本文所有路径已同步更新。

## 环境与原理
- 机器：阿里云 ECS（region=cn-shanghai），已绑定 RAM 角色（OSSReadOnlyRole，只读）。
- 鉴权：通过实例元数据服务 http://100.100.100.200 获取 STS 临时凭证（IMDSv2），
  凭证有效期约几小时，每次运行重新获取即可。
- 取数：DuckDB 的 httpfs 扩展用 S3 兼容协议直读 OSS，走内网端点（免公网流量费）。
- 依赖：Python + duckdb（若缺，用阿里云内网镜像装：
  `py -m pip install -i https://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com duckdb pyarrow pandas`）

## 连接代码模板（Python，复制即用）
```python
import json, urllib.request, duckdb

META = "http://100.100.100.200/latest"

def imds(path, token=None, method="GET", extra=None):
    headers = {}
    if token:
        headers["X-aliyun-ecs-metadata-token"] = token
    if extra:
        headers.update(extra)
    req = urllib.request.Request(META + path, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as r:
        return r.read().decode()

def get_creds():
    tok = imds("/api/token", method="PUT",
               extra={"X-aliyun-ecs-metadata-token-ttl-seconds": "1800"})
    role = imds("/meta-data/ram/security-credentials/", token=tok).strip()
    return json.loads(imds(f"/meta-data/ram/security-credentials/{role}", token=tok))

def connect():
    c = get_creds()  # 凭证字段：AccessKeyId / AccessKeySecret / SecurityToken / Expiration
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='cn-shanghai'")
    con.execute("SET s3_endpoint='oss-cn-shanghai-internal.aliyuncs.com'")  # 内网端点
    con.execute("SET s3_use_ssl=true")
    con.execute("SET s3_url_style='vhost'")
    con.execute(f"SET s3_access_key_id='{c['AccessKeyId']}'")
    con.execute(f"SET s3_secret_access_key='{c['AccessKeySecret']}'")
    con.execute(f"SET s3_session_token='{c['SecurityToken']}'")
    return con

# ---- OSS 数据路径（bucket = ys-datalake）----
BUCKET = "ys-datalake"
DAILY_BAR      = f"s3://{BUCKET}/stock/daily_bar/year=*/data.parquet"
EX_FACTOR      = f"s3://{BUCKET}/stock/ex_factor/data.parquet"            # 复权因子（单文件，无分区）
IS_ST          = f"s3://{BUCKET}/stock/is_st/year=*/data.parquet"
IS_SUSPENDED   = f"s3://{BUCKET}/stock/is_suspended/year=*/data.parquet"
SHARES         = f"s3://{BUCKET}/stock/shares/year=*/data.parquet"
MINUTE_BAR     = f"s3://{BUCKET}/stock/minute_bar/date=*/data.parquet"
OPEN_AUCTION   = f"s3://{BUCKET}/stock/open_auction/year=*/data.parquet"
FACTOR_EXPO_V2 = f"s3://{BUCKET}/stock/factor_exposure_v2/year=*/data.parquet"   # Barra CNE5 因子暴露
INSTRUMENTS    = f"s3://{BUCKET}/stock/instruments/data.parquet"
INDEX_COMP     = f"s3://{BUCKET}/index/components/year=*/data.parquet"           # 指数成分股
INDEX_WEIGHTS  = f"s3://{BUCKET}/index/weights/year=*/data.parquet"              # 指数权重
TRADING_DATES  = f"s3://{BUCKET}/common/trading_dates/data.parquet"
YIELD_CURVE    = f"s3://{BUCKET}/common/yield_curve/year=*/data.parquet"
```

## 取日频 OHLCV（最常用）
```sql
SELECT order_book_id, date, open, high, low, close, volume
FROM read_parquet(
    's3://ys-datalake/stock/daily_bar/year=*/data.parquet',
    hive_partitioning=1)
WHERE year BETWEEN 2015 AND 2026
  AND date BETWEEN '2015-01-01' AND '2026-12-31'
```

## 取分钟级数据并聚合成 K 线（参考写法）
分钟原始表 `stock/minute_bar/date=*/data.parquet`，字段含
order_book_id, datetime, open, high, low, close, volume, total_turnover, num_trades。
用 `time_bucket` 把分钟数据聚合成 N 分钟 K 线（下例为 30 分钟，标签取窗口结束，与米筐一致）：
```sql
SELECT order_book_id,
    time_bucket(INTERVAL 30 MINUTE, datetime - INTERVAL 1 MICROSECOND) + INTERVAL 30 MINUTE AS bar_time,
    arg_min(open, datetime)  AS open,
    max(high)                AS high,
    min(low)                 AS low,
    arg_max(close, datetime) AS close,
    sum(volume)              AS volume,
    sum(total_turnover)      AS total_turnover,
    sum(num_trades)          AS num_trades
FROM read_parquet('s3://ys-datalake/stock/minute_bar/date=*/data.parquet', hive_partitioning=true)
GROUP BY order_book_id, bar_time
```
- 改周期：把两个 `INTERVAL 30 MINUTE` 换成想要的窗口（如 5 MINUTE / 60 MINUTE）。
- 限定日期：加 `WHERE date BETWEEN '2024-01-01' AND '2024-01-31'`（date 是分区列，可裁剪加速）。
- 限定股票：加 `AND order_book_id IN ('600519.XSHG')`。

## OSS 数据目录速查（bucket = ys-datalake）
| 数据集 | 路径 | 说明 |
|--------|------|------|
| A股日频OHLCV | stock/daily_bar/year={2015..2026}/data.parquet | order_book_id,date,open,high,low,close,prev_close,volume,total_turnover,num_trades,limit_up,limit_down,year |
| **复权因子** | stock/ex_factor/data.parquet（单文件，无分区） | order_book_id,ex_date,ex_factor,ex_cum_factor(累计复权因子),announcement_date,ex_end_date。前复权价=raw价×ex_cum_factor[t]/ex_cum_factor[最新] |
| **ST 标记** | stock/is_st/year=*/data.parquet | order_book_id,date,is_st(BOOLEAN),year。直接判断是否 ST |
| **停牌标记** | stock/is_suspended/year=*/data.parquet | order_book_id,date,is_suspended(BOOLEAN),year。直接判断是否停牌 |
| A股分钟K线 | stock/minute_bar/date=YYYY-MM-DD/data.parquet | order_book_id,datetime,open,high,low,close,volume,total_turnover,num_trades,date |
| 开盘集合竞价 | stock/open_auction/year=*/data.parquet | order_book_id,date,open,last,high,low,prev_close,volume,total_turnover + a1-a5/b1-b5 五档挂单价与量 |
| 每日股本快照 | stock/shares/year=*/data.parquet | order_book_id,date,total,circulation_a(流通A股),non_circulation_a,total_a,preferred_shares,free_circulation(自由流通股本),year。可算换手率=volume/circulation_a、流通市值=close×circulation_a、自由流通市值=close×free_circulation |
| **Barra 因子暴露 v2** | stock/factor_exposure_v2/year=*/data.parquet | order_book_id,date + CNE5 风格因子(liquidity,leverage,earnings_variability,earnings_quality,profitability,investment_quality,book_to_price,earnings_yield,longterm_reversal,growth,momentum,mid_cap,size,beta,residual_volatility,dividend_yield,comovement) + 中信一级行业哑变量,year |
| Barra 因子暴露 v2trd | stock/factor_exposure_v2trd/year=*/data.parquet | 同上（trd 变体，行业用交易口径） |
| **指数成分股** | index/components/year=*/data.parquet | index_order_book_id,date,component_order_book_id,year。历史成分股，可做沪深300/500/1000 分池评估 |
| **指数权重** | index/weights/year=*/data.parquet | 指数成分权重 |
| 指数权重(扩展) | index/weights_ex/year=*/data.parquet | 扩展权重表 |
| 标的清单 | stock/instruments/data.parquet | order_book_id,symbol,exchange,status,special_type,board_type,industry_code/name,sector_code/name,listed_date,de_listed_date,round_lot |
| 交易日历 | common/trading_dates/data.parquet | 交易日列表 |
| 国债收益率曲线 | common/yield_curve/year=*/data.parquet | 0S~50Y 各期限无风险利率 |

## 关键注意
1. **bucket = `ys-datalake`**（原 `wengxinhua` 已删）；路径前缀 `stock/`、`common/`、`index/`（无 `quantdb/`）。
2. 读 parquet 一律加 `hive_partitioning=1`（或 true），自动识别 year=/date=/trade_date= 分区列。
3. 用分区列做 WHERE 裁剪（日频用 year、分钟用 date），少读 parquet，快很多。
4. 通配符：`daily_bar/year=*` 匹配所有年；`minute_bar/date=*` 匹配所有交易日。
5. 端点用 internal（`oss-cn-shanghai-internal.aliyuncs.com`），走内网、免流量费、低延迟。
6. 临时凭证几小时过期，每次 `connect()` 重新取，不要缓存。凭证字段为 AccessKeyId/AccessKeySecret/SecurityToken。
7. order_book_id 是米筐格式：`.XSHG`=沪市、`.XSHE`=深市（如 `600519.XSHG` 茅台）。
8. 数据覆盖：日频 2015-01-05 至今；ST/停牌/股本/复权因子 同区间；指数成分 2010 至今。
9. RAM 角色是只读，只能查不能写 OSS。

## 典型任务示例
- 取某股某段日频：`WHERE order_book_id='600519.XSHG' AND date BETWEEN '...' AND '...'`
- 取全市场某日截面：`WHERE date='2024-06-03'`
- 分钟聚合成 K 线：见上方"取分钟级数据"章节
- 换手率/流通市值（JOIN daily_bar 与 shares，按 order_book_id+date 对齐）：
```sql
SELECT d.order_book_id, d.date, d.volume, d.close,
       d.volume / NULLIF(s.circulation_a, 0)  AS turnover,         -- 换手率
       d.close * s.circulation_a              AS circ_market_cap,   -- 流通市值
       d.close * s.free_circulation           AS free_market_cap    -- 自由流通市值(中性化/规模因子用)
FROM read_parquet('s3://ys-datalake/stock/daily_bar/year=*/data.parquet', hive_partitioning=1) d
LEFT JOIN read_parquet('s3://ys-datalake/stock/shares/year=*/data.parquet', hive_partitioning=1) s
  ON d.order_book_id = s.order_book_id AND d.date = s.date
WHERE d.year=2024
```
- 日 VWAP = 成交额/成交量（前复权需乘复权因子）：
```sql
SELECT order_book_id, date,
       total_turnover / NULLIF(volume, 0) AS vwap_raw
FROM read_parquet('s3://ys-datalake/stock/daily_bar/year=*/data.parquet', hive_partitioning=1)
-- 前复权 vwap = vwap_raw × ex_cum_factor[当日] / ex_cum_factor[最新]
```
- ST/停牌过滤（用布尔标记表）：
```sql
SELECT d.* FROM read_parquet('s3://ys-datalake/stock/daily_bar/year=*/data.parquet', hive_partitioning=1) d
LEFT JOIN read_parquet('s3://ys-datalake/stock/is_st/year=*/data.parquet', hive_partitioning=1) st
  ON d.order_book_id=st.order_book_id AND d.date=st.date
LEFT JOIN read_parquet('s3://ys-datalake/stock/is_suspended/year=*/data.parquet', hive_partitioning=1) su
  ON d.order_book_id=su.order_book_id AND d.date=su.date
WHERE COALESCE(st.is_st, false)=false AND COALESCE(su.is_suspended, false)=false
```
- 算因子后存本地：结果 `to_parquet` 到本机（写不了 OSS 就写本地）

## 多进程并行取数（全史分钟聚合 → 本地 parquet）

当全史扫描（如分钟→N分钟K线聚合）耗时超 10 分钟时，瓶颈通常在 DuckDB 的 group by
（6000万+分组，hash table 远超 L3 cache，单进程 CPU 利用率仅 ~2.5/12 核）。
多进程按年份（或任意日期分区）并行可填满 CPU。

**设计要点（实测）：**
- 每个进程独立 `oss.connect()` + `PRAGMA threads=2`（agg 单进程只用 ~2.5 核，threads=2 够用不超订）
- `threads=2 × NPROC=6 = 12 核`打满；每进程只读 6 列（省 ~40% 字节）
- 纯 agg + COPY（无 LAG、无 ORDER BY），DuckDB 向量化写 parquet；无 pivot → 内存 ~3GB/进程

**性能基准（ECS 12核/47GB）：** 全史 2015-2025（11块）agg+COPY → 4GB 5min parquet：**~3min**（6进程并行）
vs 单进程串行 ~8-14min（group by CPU 不饱和）。

```python
import multiprocessing, oss

def fetch_year(args):
    year, rs, re, B, out_path = args
    con = oss.connect()
    con.execute("PRAGMA threads=2")
    con.execute(f"""
        COPY (
            SELECT order_book_id, date,
                   time_bucket(INTERVAL {B} MINUTE, datetime - INTERVAL 1 MICROSECOND) AS bt,
                   max(high) AS hi, min(low) AS lo, arg_max(close, datetime) AS cl
            FROM read_parquet('{oss.MINUTE_BAR}', hive_partitioning=1)
            WHERE date BETWEEN DATE '{rs}' AND DATE '{re}'
            GROUP BY order_book_id, date, bt
        ) TO '{out_path.replace(chr(92),"/")}' (FORMAT PARQUET);
    """)
    con.close()

# 按年份并行（每年一个 worker）
args = [(yr, f"{yr}-01-01", f"{yr}-12-31", 5, f"5min_{yr}.parquet") for yr in range(2015, 2026)]
with multiprocessing.Pool(6) as pool:
    pool.map(fetch_year, args)
```

**注意：** 多进程对 group by（CPU 不饱和）有效；对 IO bound（纯 count）无益（带宽已满）。
每进程 ~3GB 内存，NPROC=6 总 ~18GB，需确认服务器内存够。agg+COPY 落盘后后续 LAG/因子从本地 parquet 读（零 OSS 流量）。完整工具见 `fetch_5min_bench.py`。

## 复用方式
下次让 AI 帮你做因子/分析时，把本文件整段贴进去作为"取数背景"，
AI 就知道用 DuckDB + RAM 角色从 OSS（bucket=ys-datalake）取数据，不用你再解释环境。本文件自包含，不依赖其他上下文。

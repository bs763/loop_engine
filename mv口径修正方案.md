# mv 口径修正方案（log_mv 前复权 → 时点真实市值）

> 2026-08-18 ｜ 起因：`output/未来函数审计报告.md` 发现 log_mv 含未来数据
> 适用：factor_loop_engine（主）＋ 因子研究Skill（附）
> 本文档自包含，可直接交给 loop engine 侧执行

---

## 1. 问题定义

### 1.1 现状（两项目同一 SQL 惯例）

```
mv = adj_close × free_circulation
   = raw_close × f_t / L × fc        ← f_t=截至 t 的累计复权因子(时点正确)；L=全历史最新复权因子(含未来)
```

`L` 取自**数据末尾**（最新一次 ex_date 的 ex_cum_factor）→ 计算 t 日市值时用到了 t 日之后的分红/送转信息。

### 1.2 为什么错（三层）

1. **引入未来数据**：`−log L` 是每股常数、含未来公司行动，直接进截面排名。
   实测（2015-2025，5417 股）：L 分布 p50=2.68 / p95=17.09（log 跨度 ≈0~2.8）——非噪声级。
2. **不是时点市值**：前复权价 = "换算成今天股份单位的价格"，乘**当日**股本 = 单位错配。
   做过 10 送 90 的股票，送转前历史市值被低估 10 倍（越早越失真）。
3. **排名用法不需要复权**：rank_cs 逐日截面自成一体，不存在跨日比价需求；复权是给
   收益率/比率字段（除权日收益不能算错）用的，被惯例顺带套给了 mv。

### 1.3 实证影响（以 loop_engine 库内 0001 因子为例，2018-2025 全窗口 h5）

| 口径 | IC | t(NW) | 单调 | 多空年化 | 多头超额 |
|---|---|---|---|---|---|
| 现口径（qfq mv） | 0.0737 | 21.3 | 1.000 | +44.2% | +18.0% |
| 时点口径（raw mv） | ≈0.070 | ≈20.6 | 1.000 | ≈+34.6% | ≈+11.0% |

信号存活但**约 1/5 多头超额来自未来基准**。逐日截面排名相关仅 0.923，做多组（前10%）重叠 73%。

---

## 2. 修正口径（标准定义，两项目统一）

```
mv      = close × free_circulation      ← 时点真实自由流通市值（不复权，t 日全可知）
log_mv  = log(mv + 1)
adj_*   = raw × f_t / L                 ← 保持不变！收益率/比率类字段仍需前复权
```

- 拆股日：close ÷N、fc ×N → mv 连续（真实市值不变）✓
- 分红日：close 跌、fc 不变 → mv 真实下降 ✓
- 无未来信息 ✓、单位正确 ✓
- 如需总市值变体：`close × total`（shares 表有 `total` 列）；自由流通 vs 总市值是独立的建模选择，维持原选择 free_circulation
- ⚠️ 已知数据源局限（两口径同病，非本修正范围）：`free_circulation` 快照更新滞后/批量到达，
  拆股后短期内 mv 会暂时失真直到股本快照更新——OSS shares 表属性，只能接受或在字段文档标注

---

## 3. factor_loop_engine 侧修改清单

### 3.1 代码改动（仅一处）：`code/data_layer/data_loader.py`

**行 133-134**（build_factor_table 的 SELECT 内）：

```sql
-- 旧
db.close * COALESCE(exf.ex_cum_factor,1.0) / COALESCE(exlatest.cum_latest,1.0)
          * sh.free_circulation             AS mv,
-- 新
db.close * sh.free_circulation             AS mv,
```

**行 10 docstring** 同步改：

```python
# 旧
  3. 对齐 shares.free_circulation → mv = adj_close × free_circulation(**自由流通市值**,阶段1已确认用前复权 close);
# 新
  3. 对齐 shares.free_circulation → mv = close × free_circulation(**时点自由流通市值,不复权**;
     复权价只用于 adj_* 与比率字段——水平字段用复权价会引入"未来除权基准"信息,2026-08 审计修正);
```

### 3.2 文档改动：`docs/项目执行指南.md` M2 §2.3
把 mv 公式从 `adj_close × free_circulation` 改为 `close × free_circulation`，加一句修正理由（引用 §1.2）。

### 3.3 缓存失效（必做，否则改动不生效）

loop_engine 只缓存 OSS **原始表**（daily_bar/shares/is_st/is_suspended/ex_factor），mv 每次现算 →
原始缓存**不用动**。但下游字段宽表缓存把旧 mv 固化了：

```bash
rm -rf cache/panels/*.parquet      # _real_panels() 下一轮自动用新口径重建
```

⚠️ `_real_panels()` 的缓存校验只看"起始年 ≤ COMPUTE_START_YEAR"，**感知不到口径变更**——
建议顺手在校验里加个口径版本号（如文件名 `log_mv.v2.parquet` 或 panels 目录加 `MV_PIT_V1` 标记文件），防以后再改口径时踩同一个坑。

### 3.4 库内存量因子：受影响清单（16 个中 11 个）

**受影响**（值会变，指标需重测）：
| # | 表达式中的 mv/价格水平项 |
|---|---|
| 0001 | rank_cs(log_mv)、std(rank_cs(log_mv),20) |
| 0002 | rank_cs(log_mv)、roc(rank_cs(log_mv),60) |
| 0003 | roc(rank_cs(log_mv),60)、rank_cs(log_mv) |
| 0004 | rank_cs(log_mv) |
| 0005 | roc(rank_cs(log_mv),60) |
| 0009 | roc(rank_cs(log_mv),60) |
| 0012 | roc(rank_cs(log_mv),60) |
| 0013 | delta(adj_high,40) 截面 zscore（逐元素 ×1/L 不相消） |
| 0014 | roc(rank_cs(log_mv),10) |
| 0015 | rank_cs(log_mv)、roc(rank_cs(log_mv),60)、**max(adj_low,5) 截面排名** |
| 0016 | rank_cs(log_mv)、roc(rank_cs(log_mv),60) |

**不受影响**（5 个，机制免疫）：
- **0006、0008**：只用 log_amount/log_volume（原始量额，无复权）
- **0007、0010、0011**：只用比率字段（overnight/intraday/ret——同股价格相除，L 精确相消）
- 0009/0016 里的 `skew(adj_*, n)` 项本身也免疫：偏度是尺度不变量（g1=m3/m2^1.5，常数因子相消），
  但这两个因子整体仍因 mv 项受影响

### 3.5 修正后动作建议

1. `rm -rf cache/panels` → 下一轮挖掘自动重建（或手动触发 `_real_panels()`）
2. `uv run code/export_factors.py` 重新导出全库 → manifest/oos_report 指标刷新（**11 个因子指标会漂移，属预期**）
3. 决策点（由 loop 侧定）：漂移后是否重跑筛选门槛——建议只刷新档案不重筛（重筛会把 OOS 信息间接引入筛选，违反其 IS/OOS 铁律）；但**新产生的因子**自然全部走新口径
4. checkpoint.json 里存量因子的 metrics 与新导出会有出入，可在 state 里记一条口径变更标记

### 3.6 验证清单（修正后跑一遍）

```python
# ① ex-day 行为（新口径应在除权日动作更大=真实市值变化）
#    非除权日 |Δlog mv| 中位 ≈ 0.0141；除权日新口径中位 ≈ 0.0215（旧口径 0.0158，被复权抹平）
# ② log_mv 排名漂移（预期：新旧逐日截面 Spearman ≈ 0.92，做多组重叠 ≈73%——即 §1.3 的量级）
# ③ 免疫因子回归（0006/0007/0008/0010/0011 重导出后与旧 parquet 逐值相关应 = 1.0，验证改动没误伤）
# ④ 任选一个受影响因子（如 0001）重导出 + 回测，对照 §1.3 的量级（IC 0.074→~0.070）
```

---

## 4. 因子研究Skill 侧对应修改（本项目，参考执行）

本项目 `scripts/bucket_parallel.py:102-103` 同一处 SQL 同样改法；差异在于**本项目把算好的
mv 固化进了 `cache/raw/year_*.parquet`**，需缓存迁移（loop_engine 不需要）：

```python
# 迁移：year 文件里 close/free_circulation 原始列都在，本地重算 mv 列即可，无需重拉 OSS
# for each cache/raw/year_*.parquet: mv = close * free_circulation（DuckDB UPDATE 或读改写）
# 然后 rm cache/panels/base_*.parquet 重建
```

本项目两个已交付 case：0007 不受影响（比率字段）；0001 交付版按用户决策重建或保留 qfq 版本并附 PIT/raw 对照。

---

## 5. 一页速览

| 项 | 内容 |
|---|---|
| 改什么 | mv 的价格侧：前复权 close → 原始 close（`db.close * sh.free_circulation AS mv`） |
| 为什么 | 前复权基准含未来除权信息 + 单位错配 + 截面排名本不需要复权 |
| 改哪里 | loop_engine：data_loader.py L133-134（+docstring L10 + 指南 M2）；本项目：bucket_parallel.py L102-103 |
| 缓存 | loop_engine：删 cache/panels 即可；本项目：另需迁移 cache/raw 的 mv 列 |
| 影响 | 16 因子中 11 个值变（指标预期漂移 ~5-20%）；5 个免疫（比率/量额/skew 尺度不变） |
| 不改 | adj_* 列、所有比率字段口径、free_circulation 选择 |
```

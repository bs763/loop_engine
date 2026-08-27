# BRAIN 算子借鉴整理 —— 给 Claude 的执行参考

> 2026-08-28 整理 ｜ 用途：把世坤（WorldQuant BRAIN / Alpha101）的算子体系盘点一遍，
> 挑出对 factor_loop_engine 有价值、可落地的算子，供 loop 停跑后按批次实施。
> 本文档只做整理与启发，**不修改 loop 任何代码**；落地由 Claude 在 loop 停跑时执行。

---

## 0. 现状基线（2026-08-28 只读确认）

**字段 25 个**：
- 价量 13：`adj_close / adj_high / adj_low / overnight / intraday / amplitude / up_shadow / down_shadow / hl_ratio / ret / log_volume / log_amount / log_mv`
- 基本面一期 6（2026-08-24）：`roe / roa / profit_growth / bm / div_yield / ps`
- 基本面二期 6（2026-08-27）：`op_margin / asset_turn / ocf_asset / ocf_margin / debt_ratio / np_margin`（均为季更 PIT 阶梯）

**现有算子 14 个**：
- 时序 8：`ma / std / max / min / roc / delta / skew / rank_ts`
- 逐元素 4：`add / sub / mul / div`
- 截面 2：`zscore / rank_cs`

**引擎约束**（评估可借鉴性的硬条件）：
- 表达式深度 ≤ 4；窗口从 `WINDOW_SET = [5,10,20,40,60,80,120,250]` 离散取
- 面板为宽表（index=date, columns=order_book_id），时序算子组内滚动
- 全链路零未来函数（PIT + ASOF）；字段层暂无行业/分组数据
- 回测端（alphalab）已做 winsorize + standardize + 行业/市值中性化 —— 表达式层不必重复

---

## 1. BRAIN 算子全量盘点（按类）

### 1.1 截面算子（跨股票，逐日）

| BRAIN 算子 | 语义 | 我们现状 | 可借鉴性 |
|---|---|---|---|
| `rank(x)` | 截面升序排名 [0,1] | ✅ `rank_cs` | — |
| `zscore(x)` | 截面标准化 | ✅ `zscore` | — |
| `normalize(x)` | 截面缩放到 sum\|x\|=1 | ❌ | 🟡 低（zscore 够用） |
| `scale(x, a)` | 缩放使 sum\|x\|=a | ❌ | 🟡 低（回测端已 neutralization） |
| `winsorize(x, std)` | 截面缩尾 | ❌ | 🟡 低（alphalab 已 winsorize） |
| `group_rank(x, g)` | 组内排名 | ❌ | ⚫ 暂不做（依赖行业/分组字段，见 §4） |
| `group_neutralize(x, g)` | 组内中性化 | ❌ | ⚫ 暂不做，同上 |
| `group_zscore(x, g)` | 组内标准化 | ❌ | ⚫ 暂不做（随 group_rank） |
| `group_backfill(x, g, N)` | 组内回填缺失 | ❌ | 🟡 低（已有覆盖率防线） |
| `indneutralize(x, g)` | 行业中性化 | ❌ | ⚫ 暂不做（依赖行业表，见 §4） |

### 1.2 时序一元算子（单字段 + 窗口）

| BRAIN 算子 | 语义 | 我们现状 | 可借鉴性 |
|---|---|---|---|
| `ts_mean` / `ts_std_dev` | 滚动均值/标准差 | ✅ `ma` / `std` | — |
| `ts_max` / `ts_min` | 滚动极值 | ✅ `max` / `min` | — |
| `ts_delta` / `ts_rank` | 差分 / 时序排名 | ✅ `delta` / `rank_ts` | — |
| `ts_zscore(x, n)` | **时序标准化 (x−ma)/std**（个股相对自身历史偏离，**≠ 截面 zscore**） | ❌ | 🟢 高，成本低 |
| `ts_sum(x, n)` | **滚动求和** | ❌ | 🟢 高，成本极低 |
| `ts_decay_linear(x, n)` | **线性衰减加权均值**（近期权重大；BRAIN 控换手主杠杆） | ❌ | 🟢 高 |
| `ts_argmax / ts_argmin(x, n)` | **窗口极值位置**（距新高/新低几天 = 信号新鲜度；对季更字段有独特语义） | ❌ | 🟢 高 |
| `ts_skewness` | 滚动偏度 | ✅ `skew` | — |
| `ts_kurtosis` | 滚动峰度 | ❌ | 🟡 中（skew 之后可选） |
| `ts_ir(x, n)` | 滚动信息比率（均值/标准差） | ❌ | 🟡 中（≈ts_zscore 变形） |
| `ts_regression_slope(x, n)` | **滚动回归斜率**（趋势强度） | ❌ | 🟢 高 |
| `delay(x, n)` | n 天前取值 | ❌ | 🟡 低（引入未来函数风险，不建议） |
| `returns(x, n)` | n 日收益 | ✅ `roc` | — |
| `signed_power(x, p)` | sign(x)·\|x\|^p | ❌ | 🟡 中（Alpha#1 用过） |

### 1.3 时序二元算子（两字段 + 窗口）—— 我们完全空白

| BRAIN 算子 | 语义 | 我们现状 | 可借鉴性 |
|---|---|---|---|
| `ts_corr(a, b, n)` | **两字段滚动相关** | ❌ | 🔴 最高（Alpha101 最高频结构；量价背离/关系类；跨量纲天然豁免） |
| `ts_cov(a, b, n)` | **两字段滚动协方差** | ❌ | 🟢 高（corr 未标准化版，Alpha#13） |
| `ts_regression(a, b, n)` | 滚动回归（斜率/截距/R²） | ❌ | 🟡 中（corr 扩展） |

### 1.4 逐元素/条件/向量（无窗口）

| BRAIN 算子 | 语义 | 我们现状 | 可借鉴性 |
|---|---|---|---|
| `add/sub/mul/div` | 四则 | ✅ 已有 | — |
| `abs / sign / log` | 基础数学 | ⚠️ 无显式算子 | 🟡 低（可并入字段/审查端） |
| `min/max(a, b)` | 逐元素取大/小 | ✅（时序版） | — |
| `if_else(cond, a, b)` | 条件分支选择 | ❌ | 🟡 中（Alpha101 常用；需布尔面板） |
| `trade_when(x, cond)` | 条件暴露（满足才持仓） | ❌ | 🟡 低（交易行为，alphalab 层管） |
| `vec_avg(a, b, c…)` | 多字段平均 | ❌ | 🟡 低（add 展开即可） |

---

## 2. 表达力缺口分析（2026-08-28 修正视角）

> 结论先行：**现有 14 个算子其实覆盖了世坤绝大部分主流结构**，真正的"结构级"表达缺口只有一个 —— `ts_corr`。
> 以下逐一排除曾被怀疑、但实际不缺的项。

### 2.1 neg（取反）—— 不需要，被 sub 覆盖

- **单因子反转**：LLM 提示已明确「反转不用刻意做成负的」——alphalab 回测端 `direction: auto` 会自动判断多空方向，`rank_cs(intraday)` 负 IC 会以反向入库。表达式层无需 neg。
- **混合因子中"一个分支反转 + 一个分支动量"**：`sub(a, b) = a − b`，右操作数贡献天然为负 —— **sub 就是隐式 neg**。

世坤模板 E 的等价物（0 个新算子，现在就能产出）：

```
# 世坤原文（模板 E：技术反转 + 基本面动量混合，低相关两簇）
0.5 * rank(-(close / open - 1)) + 0.5 * rank(ts_rank(operating_income / equity, 126))

# 我们引擎的语义等价物（roe 替代 OI/equity，120 替代 126，等权混合=标准化后相加）
sub(rank_cs(rank_ts(roe, 120)), rank_cs(intraday))
```

过引擎关卡验证：深度 3 ≥ REVIEW_MIN_DEPTH ✓ ｜ sub(dimless, dimless) 跨量纲 ✓ ｜
rank_ts 窗口 120 ∈ (5,120) ✓ ｜ 全 PIT 后视无未来函数 ✓ ｜ 无平滑/极值嵌套 ✓

### 2.2 数字常数（0.5 权重、1−x 偏移）—— 不需要，标准化吸收

世坤的 `0.5*A + 0.5*B` 等权混合，在我们引擎里等价于「各分支先 rank_cs/zscore 标准化再 add/sub」——权重信息被标准化吸收，不影响排名方向语义。任意权重/常数偏移属于锦上添花，不是缺口。

### 2.3 唯一结构缺口：ts_corr（两字段滚动相关）

`add / sub / mul / div` 都表达不了「A 与 B 的滚动相关性」这种**关系量**。
`corr(rank_cs(log_volume), ret, 20)`（量价背离）、`corr(roe, ret, 60)`（基本面×价量关系）等 Alpha101 最高频结构，无法用现有算子凑出。**这是唯一需要新增算子才能解锁的结构。**

---

## 3. 推荐引入清单（分批实施）

### 🥇 第一批：补结构缺口（二元时序，需特殊处理）
| 算子 | 窗口建议 | 理由 |
|---|---|---|
| `ts_corr(a, b, n)` | 5–120 | **唯一结构缺口**——两字段关系类；Alpha101 最高频结构（量价背离 `corr(rank(vol), ret, 10)`、基本面×价量 `corr(roe, ret, 60)`）；输出恒 dimless，跨量纲过滤天然豁免 |
| `ts_cov(a, b, n)` | 5–120 | corr 的配套（未标准化），Alpha#13 用过 |

> ⚠️ 二元时序是结构级变更：`expression.py` 的 parse/_build/random_tree、Node 窗口处理都要扩展（ts 分支现在单子节点+单窗口 → 双子节点+窗口）。**单独提交、单独测试、好回滚**。

### 🥈 第二批：增强表达空间（一元时序，改注册表即可，非缺口）
| 算子 | 窗口建议 | 理由 |
|---|---|---|
| `ts_zscore(x, n)` | 5–120 | 个股相对自身历史偏离；对季更基本面（roe/op_margin）公告跳变后的位置感特别清晰。**注意与截面 zscore 区分，勿当重复** |
| `ts_sum(x, n)` | 5–250 | `sum(log_amount,20)` 累计量能、`sum(overnight,10)` 累计隔夜，语义直接稳定 |
| `ts_decay_linear(x, n)` | 3–120 | 近期加权平滑；对公告阶梯做"消化平滑"，也是控换手的表达手段 |

### 🥉 第三批：可选（无需新数据）
| 算子 | 理由 |
|---|---|
| `ts_argmax / ts_argmin(x, n)` | "信号新鲜度"；`ts_argmax(roe, 120)` = 上次盈利峰值距今，季更字段独特语义 |
| `ts_regression_slope(x, n)` | 趋势强度 |

---

## 4. 明确不建议加的算子

| 算子 | 原因 |
|---|---|
| `delay(x, n)` | 全链路已零未来函数（PIT+ASOF），表达式层引入 delay 易制造未来函数/语义混乱 |
| `neg(x)` | 不需要——单因子反转由 alphalab `direction: auto` 吸收；混合分支反转由 `sub` 覆盖（见 §2.1） |
| `group_rank / indneutralize(x, g)` | **暂不做（用户 2026-08-28 拍板）**——虽为世坤黄金组合（`group_rank(ts_rank(signal, N), subindustry)` 通过率最高），但依赖字段层新增行业/分组数据，短期不投入；如未来引入行业字段再评估 |
| `trade_when / normalize / scale / winsorize` | 交易行为控制与回测前处理，alphalab 层已覆盖（winsorize+standardize+中性化），表达式层重复职责 |

---

## 5. 落地注意事项（Claude 执行时）

1. **时机**：loop 正在链式运行，**必须在用户喊停后**再改代码；改文件不影响正在跑的进程（每轮独立进程），但下一轮会加载新代码。
2. **联动面**（新增算子不止改一处）：

| 位置 | 要做什么 |
|---|---|
| `code/engine/operators.py` | 注册新算子 + 窗口范围 + dim 规则（TS_OPS 表） |
| `code/engine/expression.py` | `random_tree` 自动纳入（TS_OP_NAMES 动态）；**corr 需特殊处理**（ts 分支单子节点→双子节点+窗口） |
| `code/engine/review.py` | 跨量纲规则（corr 输出 dimless 豁免）、平滑嵌套分类（decay_linear 是否算平滑算子要定口径）、roc 语义闸 |
| `code/llm/mechanisms.py` | 生成 prompt 的【算子(14)】列表、算子语义说明、机制族 hint（可顺带升级基本面机制的示例表达式） |
| LLM 审查 prompt | 算子清单同步 |
| `code/tests/` | 补算子单测 + random_tree 结构测试 |

3. **缓存**：加算子**不需要重建面板缓存**（字段没变，`PANELS_VERSION` 不动），但生成 prompt 变了，新算子下一轮就会被随机生成/LLM 用到 —— 所以要一次改好测好再放行进链式循环。
4. **测试门槛**：改完跑 `pytest` 全量 + mock 模式 `run_round_cli.py --mock --n 100` 手动验证一轮，确认无崩、无覆盖率塌陷，再让真实 loop 吃新代码。
5. **基准参考**：BRAIN 通过率实测 —— 基本面 40% > 混合 12.7% > 纯技术 5.3%；"换窗口/权重/中性化造不出低相关，低相关来自不同数据源或经济逻辑"（wq-alpha-research 实证，与库里「相关性高×N」瓶颈一致）。

---

## 6. 资料来源

- [QuantML/wq-alpha-research](https://github.com/QuantML-Research/wq-alpha-research) —— BRAIN alpha 研究 skill（SKILL.md 含官方算子速查、IS 检查、实证规律）
- [angel4angelov-glitch/wq-alpha-pipeline](https://github.com/angel4angelov-glitch/wq-alpha-pipeline) —— IQC 2026 自动化流水线（工程决策参考）
- [dao-quant-research Alpha101 深度分析](https://github.com/laozdao/dao-quant-research/blob/main/articles/M06-factor-validation/M06-07-worldquant-alpha101-analysis.md) —— 101 因子算子定义与方法论
- 中金《基于 Loop Engineering 的自动化因子发现引擎》（本项目基线研报）

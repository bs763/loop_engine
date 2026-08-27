# -*- coding: utf-8 -*-
"""核心算法层【全部超参数集中地】。每条标注【出处】,便于审视与后续标定。

出处标记:
  [研报] = 中金研报 /《项目执行指南.md》明确给定(§9 速查表或各 M 模块),忠实复现,勿随意改。
  [推断] = 研报仅一句话/示例,我落成具体规则;可商榷,真实回测后可调。
  [默认] = 研报未给,纯工程默认值;最该在【真实回测接入后】标定。

⚠️ 设计哲学(指南 §2):生成端/审查端参数是【先验约束】,绝不拿回测指标去调(防 Goodhart)。
   故本文件是「有依据的先验默认」,不追求回测最优。阶段 5 是 mock 回测(假指标),
   拿 mock 调参无意义;真正标定要等用户真实回测引擎接入后,看【整条流水线】产出的因子质量
   再决定个别旋钮——而非逐项过拟合回测。

【待标定清单】(真实回测接入后优先复盘):
  - REVIEW_MIN_DEPTH(推断,最严,可能误杀浅因子)
  - PERTURB_LR / PERTURB_BANDWIDTH / PERTURB_MIN_HISTORY(默认,研报未给)
  - review 过滤1/2 的具体规则、量纲分组(见 review.py / operators.py,推断)
"""
from __future__ import annotations

# ===== [研报] 表达式与算子(§9 / M1)=====
MAX_DEPTH = 4                       # 表达式最大深度(算子层数);研报实测平均 3.6 层
# 算子窗口范围见 operators.TS_OPS(ma 3~250 / std·max·min·rank_ts 5~120 / roc·delta 3~60 / skew 10~120)——[研报 M1]

# ===== [研报] 演化预算(M3 / §9)=====
EVOLVE_BUDGET = {
    "mutate": 0.25, "crossover": 0.25, "perturb": 0.15, "random": 0.15, "llm": 0.20,
}

# ===== [研报] 参数扰动——动量(M4 / §9)=====
MOMENTUM_BETA = 0.7                 # 动量平滑系数(新信息按 30% 融入)

# ===== [默认] 参数扰动——研报未给,工程默认(待标定)=====
PERTURB_LR = 2.0                    # 基础学习率(步长缩放)
PERTURB_BANDWIDTH = 5.0             # 高斯核带宽(窗口单位的局部邻域)
PERTURB_MIN_HISTORY = 2             # 加权最小二乘最少历史点数(不足则梯度=0,不动)

# ===== [研报] FSA 频繁子树规避(M8 / §9)=====
FSA_SUPPORT_THRESHOLD = 0.15        # 支持度阈值(占比 > 此值 且 ≥ MIN_COUNT → 冻结)
FSA_MIN_COUNT = 2                   # 最少出现次数
FSA_PARAM_VARIANT_CAP = 5           # 同骨架参数变体上限

# ===== [用户 2026-08-24] 失败模式库(结构级失败经验 → 生成端回流)=====
# 与 FSA 对偶:FSA 用库存(成功端)防结构拥挤,本库用失败端防结构浪费。
# 内因口径(用户 2026-08-24):拒因仅为 #9 IC相关 / #15 同构家族 = 占位灭(在位强因子挡路,
# 属保优淘劣挑战者路径,绝不回流规避);其余(review 结构拒/回测异常/规则 1-8/12-14/16)才算死证据。
DEAD_MIN_FAILS = 10               # 内因失败 ≥ 此值且 0 成功 → 判全灭
DEAD_PROMPT_TOP_K = 12            # 生成 prompt 注入的全灭骨架条数上限
DEAD_RESAMPLE_TRIES = 2           # 演化候选命中死骨架时每槽最大重采次数(仍死则保留,不死循环)

# ===== [用户/研报§16-17] 结构先验(2026-08-17)=====
# 背景:库内 6/34 个因子共享右半树 zscore(max(min(FLD,N),N)),std(std(·)) 亦入库;
# 研报 §16 判定「统计量堆叠/极值嵌套」为隐性过拟合(信号严重滞后),§17 实证 min/max 嵌套几乎不存活。
# 落地:① review 过滤5 拒绝平滑嵌套(ma/std 直嵌 ma/std)与极值嵌套(max/min 直嵌 max/min);
#       ② 过滤15:候选任一 ≥MIN_NODES 节点的子树骨架已在 ≥CAP 个库存因子出现 → 拒
#          (FSA #10 只对整树骨架去重,半树相同的拼接结构会绕过)。
FAMILY_SUBTREE_CAP = 2            # 同构子树家族上限(库存中出现 ≥ 此值 → 新候选拒)
FAMILY_SUBTREE_MIN_NODES = 4      # 参与家族计数的子树最小节点数(3 会误伤 std(zscore(FLD),N) 这类通用件)
COVERAGE_LOCAL_RATIO_MIN = 0.6    # 覆盖率防线:任一月覆盖 < 前后各12月中位数×此值 → 拒(回测前,确定性)
SCALE_DOMINANCE = 5.0             # 分支支配简化(2026-08-18,用户校准):顶层 add/sub 分支截面
                                  # std 比 ≥ 此值 → 取支配分支。zscore(std≈1) vs rank_cs(std≈0.29)
                                  # ≈3.4x 属「同一量级、保留」;roc(rank_cs(·)) 爆炸支 9~34x 才处理

# ===== [推断] 审查最小复杂度(M6)=====
# 研报:「最小复杂度门槛(如单算子单叶子、深度 ≤2 直接拒)」——「如」字表明是示例;
# 此处取「深度 ≤2 拒 → 最低保留 3 层」。较严,会挡掉 zscore(ma(close,20)) 这类 depth=2 因子;
# 真实回测后若发现误杀过多,可下调为 2(只拒 depth≤1)。★ 待标定
REVIEW_MIN_DEPTH = 2              # 用户 2026-08-18 放宽(原 3,干旱期接受 2 层浅因子如 zscore(ma(x,20)))

# ===== [研报] M7 回测过滤阈值(§9 速查表)=====
IC_GATE = 0.03              # 1) |IC| > 0.03
ICIR_MIN = 0.30             # 14) ICIR > 0.3(用户 2026-08-12 加严)
IC_CORR_MAX = 0.70          # 9) IC 序列相关性上限(< 0.70,与已入库因子)
SHARPE_MIN = 0.5            # 4) 区间整体夏普 / 5) 最后一年夏普 > 0.5
CALMAR_MIN = 1.0            # 6) Calmar > 1.0
MONOTONICITY_MIN = 0.85     # 13) 分组单调性 > 0.85(用户 2026-08-11 加严)
LONG_EXCESS_MIN = 0.0       # 12) 多头超额年化 > 0(用户 2026-08-11 加严)
ROLLING_MONTHS = (9, 12)        # 7/8) 近 9 月 / 12 月超额 > 0(滚动,相对末日)

# ===== [用户 2026-08-17] IS/OOS 样本切分(样本外验证)=====
# 面板仍从 COMPUTE_START_YEAR=2015 连续计算(滚动语义不变);写 alphalab 的 parquet
# 分别切 IS / OOS 窗口,2023-07~2024-12 隔离带只作 OOS 的 warmup(连续计算经过),
# 绝不进入任何评测——两段评测的 warmup 数据零交集。
# 隔离带 18 个月(≈369 交易日)≥ 最深窗口链 310 交易日(ma250 ∘ roc60)。
# 铁律:OOS 指标只存档/报告,绝不回流生成端或筛选端。
IS_START, IS_END = "2018-01-01", "2023-06-30"     # 样本内:全部筛选/入库只看这里
GAP_START, GAP_END = "2023-07-01", "2024-12-31"   # 隔离带:不评测
OOS_START, OOS_END = "2025-01-01", "2025-12-31"   # 样本外:只报告
BACKTEST_YEARS = (2018, 2023)                     # 2) 「每年都过」年份范围(IS 口径)

# ===== [用户/推断] 窗口离散集合 + warmup(用户 2026-08-11 敲定)=====
# 随机生成只从 WINDOW_SET 取(规整窗口:周/月/季/半年/年),不用任意整数;
# warmup:因子计算从 COMPUTE_START_YEAR 起,给 2018 回测留 buffer(覆盖最长 250 日窗口+嵌套)。
WINDOW_SET = [5, 10, 20, 40, 60, 80, 120, 250]
COMPUTE_START_YEAR = 2015       # 因子计算起点(warmup buffer);OSS 数据从 2015 起
BACKTEST_START = "2018-01-01"   # 回测/导出窗口(只这部分喂 alphalab / 导出)
BACKTEST_END = "2025-12-31"

# ===== [用户 2026-08-27] 累计退休制(骨架开采额度按字段代际分账)=====
# 背景:保优淘劣使 FSA#10 并发口径永不触发(全历史 0 次);子树插件被超采无记忆
# (实测 zscore(std(rank_cs(FLD),N)) 历史累计 60 次/当前库仅 2)。累计口径:每次入库
# (含被替换事件)给骨架记一次开采,累计达上限→该结构对该代际永久退休。
# 代际分账(用户要求新字段有包容度):价量账继承全部历史欠账;基本面账从实际小计数起步。
FUND_FIELDS = frozenset(["roe", "roa", "profit_growth", "bm", "div_yield", "ps",
                         "op_margin", "asset_turn", "ocf_asset", "ocf_margin",
                         "debt_ratio", "np_margin"])
# 退休线可通过 .env 覆盖(用户 2026-08-27:参数化方便调整,改后无需重启链):
#   MINED_TREE_CAP_PV / MINED_TREE_CAP_FUND / MINED_SUBTREE_CAP_PV / MINED_SUBTREE_CAP_FUND
import os as _os
def _cap(name_pv: str, name_fund: str, d_pv: int, d_fund: int) -> dict:
    return {"pv": int(_os.environ.get(name_pv, d_pv)), "fund": int(_os.environ.get(name_fund, d_fund))}
MINED_TREE_CAP = _cap("MINED_TREE_CAP_PV", "MINED_TREE_CAP_FUND", 8, 12)       # 整树(历史Top=6,温和起步)
MINED_SUBTREE_CAP = _cap("MINED_SUBTREE_CAP_PV", "MINED_SUBTREE_CAP_FUND", 12, 16)  # >=4节点子树(价量Top=60/44/27/16 立即退休)

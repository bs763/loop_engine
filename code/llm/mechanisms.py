# -*- coding: utf-8 -*-
"""12 机制族(M5)+ 机制引导生成 / 审查精判 prompt 与解析。

机制引导 = 统计因子库【未覆盖】的族 → 优先从该族选原型 → 生成自然语言假设 → 转表达式。
审查精判(M6)= 抽样检查表达式边界条件是否合理,LLM 给 ACCEPT/REJECT。

依赖 engine(算子/字段语法、表达式解析、random_tree 兜底),不接触回测指标(保持生成端 Goodhart 隔离)。
provider 由调用方注入(可切换 DeepSeek/GLM/Mock)。

⚠️ 数量说明:研报正文写"13 个机制族",但图表 7 实际只列 **12 个顶层族**(8 时序 + 4 截面),
这是研报本身的小幅出入。本项目**以图表 7 的 12 族为准**(见 docs/项目执行指南.md M5),不硬凑 13。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from engine.expression import Node, parse, random_tree
from engine.operators import CS_OP_NAMES, ELEM_OP_NAMES, TS_OP_NAMES

# ============================================================================
# 族级缺陷记忆(终审拒因回流,用户 2026-08-18 设计)——防 Goodhart 安全变体:
# 只回流「结构/边界/经济」类批评(先验知识),指标类拒因只进台账,指标数值永不回流生成端。
# ============================================================================
_FAMILY_NOTES_PATH = Path("output/family_notes.json")     # 相对项目根(与 lessons 同级)
_family_notes: dict | None = None


def family_notes() -> dict:
    """{机制族id: [已知缺陷, ...]},惰性加载自 output/family_notes.json,跨轮持久。"""
    global _family_notes
    if _family_notes is None:
        try:
            _family_notes = json.loads(_FAMILY_NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  首次/损坏 → 空
            _family_notes = {}
    return _family_notes


def add_family_note(mech_id: str, note: str, cap: int = 5) -> None:
    """记录一条该机制族的已知缺陷(去重,每族最多 cap 条),立即落盘。"""
    notes = family_notes().setdefault(mech_id, [])
    if note not in notes:
        notes.append(note)
        del notes[:-cap]
        _FAMILY_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FAMILY_NOTES_PATH.write_text(json.dumps(family_notes(), ensure_ascii=False, indent=1),
                                      encoding="utf-8")


# 终审拒因分类:含指标词汇 → 指标类(不回流);否则 → 结构/经济类(可回流生成端)
_METRIC_HINTS = ("IC", "ic", "夏普", "ICIR", "年化", "收益", "单调", "超额", "回撤",
                 "almar", "年份", "多头", "多空", "turnover", "胜率")


def is_metric_reason(reason: str) -> bool:
    return any(h in reason for h in _METRIC_HINTS)


# 表达式 hash → 生成它的机制族 id(进程内登记;生成与终审同进程,足够)
_expr_family: dict[str, str] = {}


def family_of(expr_hash: str) -> str | None:
    return _expr_family.get(expr_hash)


def _register_family(expr_hash: str, mech_id: str, cap: int = 4000) -> None:
    _expr_family[expr_hash] = mech_id
    if len(_expr_family) > cap:
        _expr_family.pop(next(iter(_expr_family)))

# ============================================================================
# 12 机制族(时序 8 + 截面 4)——以图表 7 为准(见模块 docstring)
# ============================================================================
MECHANISMS: list[dict] = [
    # ---- 时序类(8)----
    {"id": "ts_trend_momentum", "category": "ts", "name": "趋势与动量",
     "prototypes": ["趋势启动", "趋势延续", "动量衰竭"],
     "hint": "价格/收益的持续性:过去 N 日收益或均线斜率具有惯性,延续或衰竭。",
     "field_hints": ["ret", "adj_close", "overnight"]},
    {"id": "ts_reversal", "category": "ts", "name": "反转与均值回归",
     "prototypes": ["短期过度反应", "抛压耗尽后修复", "突破失败后反转"],
     "hint": "短期极端收益后的均值回归;过度下跌/抛压释放后的修复。",
     "field_hints": ["ret", "overnight", "hl_ratio"]},
    {"id": "ts_breakout", "category": "ts", "name": "边界突破",
     "prototypes": ["上沿突破并站稳", "下沿跌破并延续", "回踩确认后再启动"],
     "hint": "价格突破近期高/低边界后的延续,或回踩确认。",
     "field_hints": ["adj_high", "adj_low", "adj_close", "hl_ratio"]},
    {"id": "ts_candle", "category": "ts", "name": "K 线与日内结构",
     "prototypes": ["实体强度与收盘位置", "影线压力与价格拒绝", "多日转折结构"],
     "hint": "K 线形态信号:实体强弱、上下影线代表的价格拒绝、收盘区间位置。",
     "field_hints": ["up_shadow", "down_shadow", "hl_ratio", "intraday"]},
    {"id": "ts_overnight_gap", "category": "ts", "name": "跳空与隔夜定价",
     "prototypes": ["缺口延续", "缺口回补", "隔夜与日内方向切换"],
     "hint": "隔夜收益与日内收益的方向切换、跳空缺口的延续或回补。",
     "field_hints": ["overnight", "intraday", "adj_open"]},
    {"id": "ts_vol_compress", "category": "ts", "name": "波动压缩与释放",
     "prototypes": ["波动压缩", "扩张后修复", "区间能量积累与释放"],
     "hint": "波动率压缩(低波动)后的扩张释放;区间能量积蓄后的突破。",
     "field_hints": ["amplitude", "ret"]},
    {"id": "ts_vol_state", "category": "ts", "name": "波动状态",
     "prototypes": ["高波动状态信号", "低波动状态信号", "波动状态切换"],
     "hint": "当前处于高/低波动状态,或波动状态发生切换的信号。",
     "field_hints": ["amplitude", "hl_ratio"]},
    {"id": "ts_drawdown", "category": "ts", "name": "回撤与修复",
     "prototypes": ["渐进式回撤", "深度回撤后恢复", "反弹失败后二次下跌"],
     "hint": "价格回撤深度与修复进度;深度回撤后的反弹或反弹失败再下跌。",
     "field_hints": ["ret", "adj_close"]},
    # ---- 截面类(4)----
    {"id": "cs_relative_strength", "category": "cs", "name": "相对强弱与定价偏离",
     "prototypes": ["市场相对动量", "同期极端收益反转", "相对价格位置偏离"],
     "hint": "截面上个股相对市场的强弱、极端收益反转、相对价格位置偏离。",
     "field_hints": ["ret", "adj_close"]},
    {"id": "cs_risk_return", "category": "cs", "name": "截面风险收益错配",
     "prototypes": ["低波动中的相对强势", "高波动但收益补偿不足", "收益与尾部风险排序背离"],
     "hint": "截面上风险与收益的错配:低波动者的相对强势、高波动补偿不足等。",
     "field_hints": ["ret", "amplitude", "log_mv"]},
    {"id": "cs_vol_amplitude", "category": "cs", "name": "截面波动与振幅异常",
     "prototypes": ["相对低波动溢价", "异常振幅扩张", "波动排名突变"],
     "hint": "截面上相对波动/振幅的异常:低波动溢价、振幅突然扩张、波动排名突变。",
     "field_hints": ["amplitude", "up_shadow", "down_shadow"]},
    {"id": "cs_dispersion", "category": "cs", "name": "截面分化与一致性",
     "prototypes": ["收益分化扩张", "波动分化加剧", "市场一致性上升"],
     "hint": "截面上收益/波动的分化程度或市场一致性变化。",
     "field_hints": ["ret", "amplitude"]},
]

FIELD_MEANINGS = ("open/high/low/close=价, volume=成交量, amount=成交额, "
                  "overnight=隔夜收益, intraday=日内收益, amplitude=振幅, "
                  "up_shadow/down_shadow=影线占比, hl_ratio=收盘区间位, ret=收益, "
                  "log_volume/log_amount/log_mv=对数规模")

# 机制族 boost:把 LLM 生成额外拉向「隔夜跳空」等未充分挖掘机制族(权重乘数)
MECHANISM_BOOST: dict[str, float] = {
    "ts_overnight_gap": 4.0,    # 跳空与隔夜定价(核心空类)
    "ts_candle": 2.0,           # K 线与日内结构(影线/intraday)
    "ts_vol_compress": 2.0,     # 波动压缩与释放(amplitude)
    "ts_vol_state": 2.0,        # 波动状态(amplitude/hl_ratio)
}


# ============================================================================
# Prompt 构造
# ============================================================================

def _grammar() -> str:
    return (f"- 时序(带窗口 n): {', '.join(TS_OP_NAMES)},例: ma(close, 20)\n"
            f"- 逐元素(两参数): {', '.join(ELEM_OP_NAMES)},例: sub(high, low)\n"
            f"- 截面(一参数): {', '.join(CS_OP_NAMES)},例: zscore(close)")


def build_generation_prompt(mech: dict, fields: list[str]) -> str:
    notes = family_notes().get(mech.get("id", ""), [])
    avoid = ("\n【该机制族的已知缺陷,生成时务必规避】\n- " + "\n- ".join(notes)) if notes else ""
    return f"""你是 A 股量化研究员。基于下列因子语法,生成【恰好一个】因子表达式,体现指定市场机制。

【算子(14)】
{_grammar()}
【可用字段】: {', '.join(fields)}
【硬规则】
- s-表达式嵌套,最大深度 4 层;
- add/sub 不得跨量纲(禁止 add(close, volume) 这类);只用上面字段;窗口为正整数。
【目标机制】{mech['name']}: {', '.join(mech['prototypes'])}
【经济学假设】{mech['hint']}{avoid}

只输出一个合法 s-表达式,不要解释、不要 markdown。"""


def build_review_prompt(node: Node, metrics=None) -> str:
    """终审 prompt。metrics 可携带 IS 回测指标(选择端可见,防 Goodhart 不禁)——
    仅供诊断(单年依赖/多头无肉等规则盲区),不得仅因指标高低拒收(门槛由 16 项规则负责)。"""
    mblock = ""
    if metrics is not None:
        g = (metrics.get if isinstance(metrics, dict)
             else lambda k: getattr(metrics, k, None))
        annual = (metrics.get("annual_ls_return") if isinstance(metrics, dict)
                  else getattr(metrics, "annual_ls_return", None)) or {}
        yr = ("  逐年多空: " + ", ".join(f"{y}:{v:+.1%}" for y, v in sorted(annual.items()))
              if annual else "")
        mblock = f"""
【IS 回测指标(仅诊断参考;不得仅因指标高低而拒/收——指标门槛由 16 项规则负责)】
IC={g('ic_mean'):+.4f} ICIR={g('icir'):.2f} 夏普={g('ls_sharpe'):.2f} Calmar={g('calmar'):.2f}
多头超额年化={g('long_excess_annual'):+.2%} 单调性={g('monotonicity'):.2f}{yr}
→ 先逐项核对逐年多空再裁决:①若最大单年收益超过其余年份总和(收益集中于单一年份),
  必须拒;②若半数以上年份收益接近零(如 |y|<2%),倾向拒(单年依赖);③若多头超额
  年化<1% 而多空尚可(空头独撑),倾向拒。核对结论必须写进理由。"""
    return f"""你是 A 股量化研究员,审查因子表达式的边界合理性。

【表达式】{node.to_str()}
【字段含义】{FIELD_MEANINGS}
【算子语义】时序算子(ma/std/max/min/roc/delta/skew/rank_ts)的第二参数是**滚动窗口天数**,
如 max(x, 40) = x 的 40 日滚动最大值——**不是数值截断**;zscore/rank_cs 为逐截面(跨股票)算子。{mblock}

请检查:① 边界条件(除零、极端窗口、量纲错配);② 经济学含义是否自洽;
③ 过度平滑(仅限两种确定性口径,其它一律不算违规):平滑算子的**直接子节点**也是平滑算子
   (std/std、ma/ma、ma/std 等「平滑套平滑」),或极值算子的**直接子节点**也是极值算子
   (max/min、max/max 等「极值套极值」)。中间隔了其它算子的不算——如 max(skew(·),N)、
   std(rank_cs(·),N)、ma(zscore(std(·),N)) 均为正常组合,不得据此拒;
   结构先验由代码层规则精确执行,你的重点是①边界与②经济学含义。
只输出一行:ACCEPT 或 REJECT,后接一句理由(例:REJECT: div 分母可能为零)。"""


# ============================================================================
# LLM 输出解析
# ============================================================================

def _first_balanced_sexpr(text: str) -> str | None:
    """从文本中找第一个 `name(...)` 平衡括号子串。"""
    for m in re.finditer(r"([a-zA-Z_]\w*)\s*\(", text):
        depth = 0
        for i in range(m.end() - 1, len(text)):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return text[m.start():i + 1]
    return None


def extract_expression(text: str | None, allowed_fields: list[str] | None = None) -> Node | None:
    """从 LLM 文本抽取合法表达式 Node;非法或字段越界返回 None。容错 fenced/带解释。"""
    if not text:
        return None
    m = re.search(r"```(?:[a-zA-Z]*)?\s*(.*?)\s*```", text, re.S)  # 先取代码块
    candidate = m.group(1) if m else text
    s = _first_balanced_sexpr(candidate)
    if s is None:  # 退化:裸叶子
        leaf = re.search(r"\b([a-zA-Z_]\w+)\b", candidate)
        s = leaf.group(1) if leaf else None
    if not s:
        return None
    try:
        node = parse(s)
        node.validate()
    except Exception:
        return None
    if allowed_fields and not node.fields().issubset(set(allowed_fields)):
        return None
    return node


def parse_verdict(text: str | None) -> tuple[bool, str]:
    """解析审查裁决 → (accept, reason)。无法解析→默认放行(True)。"""
    t = (text or "").strip()
    upper = t.upper()
    if re.search(r"\bACCEPT\b", upper):
        return True, t
    if re.search(r"\bREJECT\b", upper):
        return False, t
    return True, t


# ============================================================================
# 机制选择 / 生成 / 审查
# ============================================================================

def pick_mechanism(mechanisms: list[dict] | None = None,
                   coverage: dict[str, int] | None = None,
                   rng: np.random.Generator | None = None,
                   boost: dict[str, float] | None = None) -> dict:
    """选机制族;给 coverage 时按 boost/(1+count) 加权,优先【未覆盖】族(M5)并额外拉向 boost 族。"""
    mechs = mechanisms or MECHANISMS
    rng = rng or np.random.default_rng()
    if not coverage:
        return mechs[int(rng.integers(0, len(mechs)))]
    b = boost or {}
    weights = np.array([b.get(m["id"], 1.0) / (1 + coverage.get(m["id"], 0)) for m in mechs])
    weights = weights / weights.sum()
    return mechs[int(rng.choice(len(mechs), p=weights))]


def _bump(provider, key: str) -> None:
    """在 provider 实例上累计 LLM 调用统计(生成/审查健康度;run_round_cli 每轮随 STATUS 汇报)。"""
    setattr(provider, key, getattr(provider, key, 0) + 1)


def generate_expression(provider, fields: list[str], mechanisms: list[dict] | None = None,
                        rng: np.random.Generator | None = None,
                        max_retries: int = 1, temperature: float = 0.8,
                        field_usage: dict[str, int] | None = None,
                        boost: dict[str, float] | None = None) -> Node:
    """机制引导生成:选族(偏向未充分挖掘的机制 + boost 优先族)→ prompt → 解析;非法重试,全失败→random_tree 兜底。"""
    rng = rng or np.random.default_rng()
    coverage = None
    if field_usage:
        mechs = mechanisms or MECHANISMS
        coverage = {m["id"]: sum(field_usage.get(f, 0) for f in m.get("field_hints", []))
                    for m in mechs}
    mech = pick_mechanism(mechanisms, coverage=coverage, rng=rng, boost=boost)
    prompt = build_generation_prompt(mech, fields)
    for _ in range(max_retries + 1):
        try:
            text = provider.complete(prompt, temperature=temperature)
        except Exception:
            _bump(provider, "llm_gen_api_error")
            continue  # LLM 超时/连接错误 → 重试(全失败走兜底,不崩轮)
        node = extract_expression(text, allowed_fields=fields)
        if node is not None and not node.is_leaf():
            _bump(provider, "llm_gen_ok")
            _register_family(node.expr_hash(), mech["id"])   # 供终审拒因回流定位族
            return node
        _bump(provider, "llm_gen_bad_output")
    _bump(provider, "llm_gen_fallback")
    return random_tree(fields, rng=rng)  # 全失败兜底(API 错或输出非法)


def review_expression(provider, node: Node, temperature: float = 0.1,
                      metrics=None) -> tuple[bool, str]:
    """LLM 审查精判:返回 (accept, reason)。metrics=IS 回测指标(仅诊断参考,选择端合法)。
    LLM 超时/连接错 → 放行(不崩)。"""
    prompt = build_review_prompt(node, metrics=metrics)
    try:
        text = provider.complete(prompt, temperature=temperature)
    except Exception:
        _bump(provider, "llm_rev_error")
        return True, "LLM 审查不可用,放行"
    _bump(provider, "llm_rev_ok")
    return parse_verdict(text)


def make_evolve_llm_hook(provider, rng: np.random.Generator | None = None,
                         boost: dict[str, float] | None = None):
    """适配 evolve.Evolver.llm_provider 的 callable:(tree, fields, rng, field_usage) -> Node。
    boost 默认用 MECHANISM_BOOST(跳空优先)。"""
    b = MECHANISM_BOOST if boost is None else boost

    def hook(tree, fields, r, field_usage=None):
        return generate_expression(provider, list(fields), rng=r if r is not None else rng,
                                   field_usage=field_usage, boost=b)
    return hook

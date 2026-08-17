# -*- coding: utf-8 -*-
"""mechanisms.py 单元测试:13 机制族、prompt、抽取容错、生成/审查/回退、hook。"""
import numpy as np
import pytest

from engine.expression import parse
from llm import mechanisms as M
from llm.provider import MockProvider

ALLOWED = ["close", "high", "low", "volume", "overnight", "amplitude"]


# ---------------- 13 机制族数据 ----------------

def test_mechanisms_count_and_categories():
    # 图表 7:12 族 = 8 时序 + 4 截面(研报正文"13"为出入,以图表 7 为准)
    assert len(M.MECHANISMS) == 12
    ts = [m for m in M.MECHANISMS if m["category"] == "ts"]
    cs = [m for m in M.MECHANISMS if m["category"] == "cs"]
    assert len(ts) == 8 and len(cs) == 4


def test_mechanisms_well_formed():
    ids = [m["id"] for m in M.MECHANISMS]
    assert len(set(ids)) == 12  # id 唯一
    for m in M.MECHANISMS:
        assert {"id", "category", "name", "prototypes", "hint", "field_hints"} <= set(m)
        assert len(m["prototypes"]) >= 1


def test_no_inferred_family():
    # 不硬凑第 9 时序族;12 族全部来自图表 7
    assert all(not m.get("_inferred") for m in M.MECHANISMS)


# ---------------- Prompt ----------------

def test_review_prompt_includes_oversmoothing():
    # 审查第③类:过度平滑(2026-08-17 收窄口径——仅平滑套平滑/极值套极值,
    # 其它堆叠(如 max(skew)、std(rank_cs))明确不得据此拒,防止 LLM 比代码规则更宽而误杀)
    p = M.build_review_prompt(parse("zscore(ma(close, 20))"))
    assert "过度平滑" in p and "极值套极值" in p and "不得据此拒" in p


def test_generation_prompt_has_essentials():
    mech = M.MECHANISMS[0]
    p = M.build_generation_prompt(mech, ALLOWED)
    assert mech["name"] in p
    assert "close" in p and "overnight" in p
    assert "深度" in p


def test_review_prompt_has_expression_and_verdict_keywords():
    node = parse("zscore(ma(close, 20))")
    p = M.build_review_prompt(node)
    assert "zscore(ma(close, 20))" in p
    assert "ACCEPT" in p and "REJECT" in p


# ---------------- 抽取容错 ----------------

def test_extract_raw_expression():
    n = M.extract_expression("zscore(ma(close, 20))", ALLOWED)
    assert n is not None and n.to_str() == "zscore(ma(close, 20))"


def test_extract_fenced_block():
    n = M.extract_expression("结果是:\n```\ndiv(high, low)\n```\n如上。", ALLOWED)
    assert n is not None and n.to_str() == "div(high, low)"


def test_extract_with_surrounding_explanation():
    n = M.extract_expression("我建议用 sub(high, low) 表达影线。", ALLOWED)
    assert n is not None and n.to_str() == "sub(high, low)"


def test_extract_invalid_returns_none():
    assert M.extract_expression("这根本不是表达式 blah blah", ALLOWED) is None
    assert M.extract_expression("add(close)", ALLOWED) is None  # arity 错


def test_extract_rejects_field_outside_allowed():
    # mv 不在 ALLOWED → 拒
    assert M.extract_expression("zscore(ma(mv, 20))", ALLOWED) is None
    # 不传 allowed_fields → 放行(未知字段也解析)
    assert M.extract_expression("zscore(ma(mv, 20))") is not None


# ---------------- 裁决 ----------------

def test_parse_verdict():
    assert M.parse_verdict("ACCEPT: 合理")[0] is True
    assert M.parse_verdict("REJECT: 除零风险")[0] is False
    assert M.parse_verdict("看不懂") == (True, "看不懂")  # 默认放行


# ---------------- 生成 / 审查 / 回退 ----------------

def test_generate_valid_expr():
    prov = MockProvider(response="zscore(div(ma(close, 20), std(high, 10)))")
    node = M.generate_expression(prov, ALLOWED, rng=np.random.default_rng(0))
    node.validate()
    assert not node.is_leaf()
    assert node.fields().issubset(set(ALLOWED))


def test_generate_falls_back_on_garbage():
    prov = MockProvider(response="完全不是表达式")
    node = M.generate_expression(prov, ALLOWED, rng=np.random.default_rng(1))
    node.validate()  # 回退到 random_tree,仍合法
    assert node.fields().issubset(set(ALLOWED))


def test_generate_falls_back_on_field_violation():
    prov = MockProvider(response="zscore(ma(mv, 20))")  # mv 不允许
    node = M.generate_expression(prov, ALLOWED, rng=np.random.default_rng(2))
    node.validate()
    assert node.fields().issubset(set(ALLOWED))  # 回退后字段合规


def test_generate_falls_back_on_api_error():
    # provider.complete 抛异常(超时/连接错)→ 不崩,回退 random_tree
    class _FailProvider:
        def complete(self, prompt, system=None, temperature=None):
            raise ConnectionError("read timeout")
    node = M.generate_expression(_FailProvider(), ALLOWED, rng=np.random.default_rng(3))
    node.validate()
    assert node.fields().issubset(set(ALLOWED))


def test_review_intercepts():
    prov = MockProvider(response="REJECT: div 分母可能为零")
    accept, reason = M.review_expression(prov, parse("div(close, volume)"))
    assert accept is False
    assert "REJECT" in reason


def test_review_accepts():
    prov = MockProvider(response="ACCEPT: 边界合理")
    accept, _ = M.review_expression(prov, parse("zscore(ma(close, 20))"))
    assert accept is True


# ---------------- 机制选择 ----------------

def test_pick_mechanism_returns_member():
    rng = np.random.default_rng(3)
    m = M.pick_mechanism(rng=rng)
    assert m in M.MECHANISMS


def test_pick_mechanism_avoids_over_covered():
    rng = np.random.default_rng(4)
    coverage = {"ts_trend_momentum": 1000}  # 该族被大量覆盖
    picked = [M.pick_mechanism(coverage=coverage, rng=rng)["id"] for _ in range(100)]
    # 高覆盖族应被显著回避(< 10%)
    assert picked.count("ts_trend_momentum") < 10


# ---------------- evolve hook ----------------

def test_make_evolve_llm_hook():
    prov = MockProvider(response="zscore(ma(close, 20))")
    hook = M.make_evolve_llm_hook(prov, rng=np.random.default_rng(5))
    node = hook(tree=None, fields=ALLOWED, r=None)
    node.validate()
    assert node.to_str() == "zscore(ma(close, 20))"

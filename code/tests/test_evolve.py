# -*- coding: utf-8 -*-
"""evolve.py 单元测试:保结构、预算分布、冷启动、各操作正确性。"""
import numpy as np
import pytest

from engine.expression import parse, random_tree, MAX_DEPTH
from engine.evolve import Evolver, EvolveConfig, DEFAULT_BUDGET
from engine.perturb import Perturber

FIELDS = ["close", "high", "low", "volume", "overnight", "intraday"]
PARENTS = [
    parse("zscore(sub(ma(close, 20), ma(low, 20)))"),
    parse("div(ma(overnight, 60), std(high, 30))"),
    parse("rank_cs(sub(ma(volume, 10), ma(close, 10)))"),
]


def _new(seed=0, perturber=None, llm=None):
    return Evolver(FIELDS, rng=np.random.default_rng(seed), perturber=perturber, llm_provider=llm)


def _all_valid(trees, fields):
    for t in trees:
        t.validate()
        assert t.depth() <= MAX_DEPTH
        assert t.fields().issubset(set(fields))


# ---------------- 保结构 ----------------

def test_generate_cold_start_valid():
    e = _new(1)
    trees = e.generate([], 200)
    assert len(trees) == 200
    _all_valid(trees, FIELDS)


def test_generate_with_parents_valid():
    e = _new(2)
    trees = e.generate(PARENTS, 300)
    _all_valid(trees, FIELDS)


def test_mutate_valid():
    e = _new(3)
    for _ in range(200):
        t = e.mutate(PARENTS[0])
        t.validate()
        assert t.depth() <= MAX_DEPTH
        assert t.fields().issubset(set(FIELDS))


def test_crossover_bounded_depth():
    e = _new(4)
    for _ in range(300):
        t = e.crossover(PARENTS[0], PARENTS[1])
        t.validate()
        assert t.depth() <= MAX_DEPTH


def test_perturb_preserves_structure():
    e = _new(5)
    for parent in PARENTS:
        t = e.perturb_op(parent)
        t.validate()
        assert t.depth() <= MAX_DEPTH
        # 结构(算子骨架)不变,只可能改窗口
        assert t.to_str().split("(")[0] == parent.to_str().split("(")[0]


def test_perturb_cold_leaves_windows_unchanged():
    """冷启动 perturber 无历史 → propose 返回原值,窗口不变。"""
    e = _new(6, perturber=Perturber())
    parent = parse("ma(ma(close, 20), 5)")
    t = e.perturb_op(parent)
    assert t.to_str() == "ma(ma(close, 20), 5)"


# ---------------- 预算分布 ----------------

def test_pick_op_distribution():
    e = _new(7)
    counts = {"mutate": 0, "crossover": 0, "perturb": 0, "random": 0, "llm": 0}
    for _ in range(5000):
        counts[e._pick_op()] += 1
    for op, budget in DEFAULT_BUDGET.items():
        assert abs(counts[op] / 5000 - budget) < 0.03, f"{op}: {counts[op]/5000:.3f} vs {budget}"


def test_config_budget_must_sum_to_one():
    with pytest.raises(ValueError):
        EvolveConfig(mutate=0.5, crossover=0.5, perturb=0.1, random=0.1, llm=0.1)


# ---------------- 冷启动 / LLM stub ----------------

def test_cold_start_no_crash_and_diverse():
    e = _new(8)
    trees = e.generate([], 50)
    strs = {t.to_str() for t in trees}
    assert len(strs) > 1  # 随机生成有多样性


def test_llm_stub_without_provider():
    e = _new(9, llm=None)
    t = e.llm_op()
    t.validate()
    assert t.fields().issubset(set(FIELDS))


def test_llm_provider_hook_called():
    """传 provider 时,llm_op 走 provider。"""
    called = {"n": 0}

    def provider(tree, fields, rng, field_usage=None):
        called["n"] += 1
        return parse("zscore(ma(close, 10))")

    e = _new(10, llm=provider)
    t = e.llm_op(PARENTS[0])
    assert called["n"] == 1
    assert t.to_str() == "zscore(ma(close, 10))"


def test_generate_skips_over_depth_and_tops_up():
    """LLM 偶发产出 depth>4 → 跳过,不崩,random_tree 补足 n 个合法。"""
    from engine.evolve import EvolveConfig
    from llm.mechanisms import make_evolve_llm_hook
    from llm.provider import MockProvider
    deep = "ma(ma(ma(ma(ma(close, 5), 5), 5), 5), 5)"  # depth 5(5 层 ma 嵌套)
    assert parse(deep).depth() == 5
    cfg = EvolveConfig(mutate=0, crossover=0, perturb=0, random=0, llm=1.0)  # 全走 LLM
    e = Evolver(FIELDS, config=cfg, rng=np.random.default_rng(1),
                llm_provider=make_evolve_llm_hook(MockProvider(response=deep)))
    trees = e.generate(PARENTS, 5)
    assert len(trees) == 5                       # 仍产出 5 个
    for t in trees:
        t.validate()
        assert t.depth() <= MAX_DEPTH            # 全 ≤4(depth-5 的被跳过)

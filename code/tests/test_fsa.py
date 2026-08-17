# -*- coding: utf-8 -*-
"""fsa.py 单元测试:骨架抽象、冻结、参数变体上限、状态持久化。"""
from engine.expression import parse
from engine.fsa import skeleton, FSA


# ---------------- 骨架抽象 ----------------

def test_skeleton_basic():
    assert skeleton(parse("close")) == "FLD"
    assert skeleton(parse("sub(high, low)")) == "sub(FLD, FLD)"
    assert skeleton(parse("ma(close, 20)")) == "ma(FLD, N)"
    assert skeleton(parse("zscore(close)")) == "zscore(FLD)"


def test_skeleton_ignores_field_and_window():
    # 不同字段/窗口、同结构 → 同骨架
    a = skeleton(parse("div(ma(close, 20), ma(close, 10))"))
    b = skeleton(parse("div(ma(volume, 118), ma(amount, 66))"))
    assert a == "div(ma(FLD, N), ma(FLD, N))"
    assert a == b


# ---------------- 冻结 ----------------

def test_freeze_when_dominant():
    fsa = FSA()
    a = "div(ma(FLD, N), ma(FLD, N))"   # 占多数
    for _ in range(3):
        fsa.observe(a)
    fsa.observe("zscore(FLD)")
    # a 占 3/4=75% > 15%,次数 3≥2 → 冻结
    assert fsa.is_frozen(a)
    assert not fsa.is_frozen("zscore(FLD)")
    ok, reason = fsa.check(parse("div(ma(close, 5), ma(close, 10))"))
    assert not ok and "frozen" in reason


def test_no_freeze_when_rare():
    fsa = FSA()
    # 多样化库,每个骨架只 1 次 → 不冻结
    for sk in ["a(FLD)", "b(FLD)", "c(FLD)", "d(FLD)", "e(FLD)", "f(FLD)"]:
        fsa.observe(sk)
    assert fsa.total() == 6
    assert not fsa.is_frozen("a(FLD)")  # 1/6≈16.7%? 次数 1 < min_count=2 → 不冻结
    # 注:1 次 < min_count=2,即便占比高也不冻结


# ---------------- 参数变体上限 ----------------

def test_param_variant_cap():
    fsa = FSA()
    a = parse("ma(close, 5)")           # 骨架 ma(FLD, N)
    skel_a = "ma(FLD, N)"
    # 让 a 骨架占比低(不冻结):加大量其他骨架
    for _ in range(5):
        fsa.observe(skel_a)
    for i in range(40):
        fsa.observe(f"other{i}(FLD, N)")
    # a 骨架 5/45≈11% <15%(不冻结),但已达 5 个变体上限
    assert not fsa.is_frozen(skel_a)
    ok, reason = fsa.check(a)
    assert not ok and "param_variant_cap" in reason


def test_allows_within_cap():
    fsa = FSA()
    fsa.observe("ma(FLD, N)")
    fsa.observe("ma(FLD, N)")
    # 加足量其他骨架,把 ma(FLD,N) 占比压到 15% 以下,避免冻结
    for i in range(15):
        fsa.observe(f"other{i}(FLD)")
    ok, reason = fsa.check(parse("ma(close, 30)"))
    assert ok, f"应通过但被拒: {reason}"  # 2 个变体 < 上限5,占比 2/17≈11.8% <15% 未冻结


# ---------------- 状态持久化 ----------------

def test_state_roundtrip():
    fsa = FSA()
    fsa.observe("ma(FLD, N)")
    fsa.observe("ma(FLD, N)")
    fsa.observe("zscore(FLD)")
    snap = fsa.state()

    fsa2 = FSA()
    fsa2.load_state(snap)
    assert fsa2.counts == fsa.counts
    assert fsa2.support("ma(FLD, N)") == fsa.support("ma(FLD, N)")

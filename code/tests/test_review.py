# -*- coding: utf-8 -*-
"""review.py 单元测试:四道过滤(折叠截面 / 同质退化 / 跨量纲 / 最小复杂度)。"""
from engine.expression import Node, parse
from engine.review import apply, simplify, MIN_DEPTH


# ---------------- 过滤1:截面折叠(简化) ----------------

def test_simplify_nested_cs():
    t = parse("zscore(zscore(ma(close, 20)))")
    assert simplify(t).to_str() == "zscore(ma(close, 20))"
    # 混合截面嵌套也折叠(去外层保内层)
    t2 = parse("rank_cs(zscore(ma(close, 20)))")
    assert simplify(t2).to_str() == "zscore(ma(close, 20))"


def test_simplify_no_change_when_not_nested():
    t = parse("zscore(ma(close, 20))")
    assert simplify(t).to_str() == "zscore(ma(close, 20))"


# ---------------- 过滤4:最小复杂度 ----------------

def test_reject_shallow():
    assert apply(parse("close"))[0] is None                      # depth 0
    assert apply(parse("ma(close, 20)"))[0] is None              # depth 1
    assert apply(parse("zscore(close)"))[0] is None              # depth 1
    assert apply(parse("zscore(ma(close, 20))"))[0] is None      # depth 2


def test_accept_min_depth():
    # depth 3,同量纲(price−price),通过
    t, reason = apply(parse("zscore(sub(ma(close, 20), ma(close, 10)))"))
    assert t is not None, reason


# ---------------- 过滤2:同质退化 ----------------

def test_reject_degenerate_subtree():
    # depth 3,但子树 sub(ma(close,20), ma(close,20)) 两子相同 → 退化
    t, reason = apply(parse("zscore(sub(ma(close, 20), ma(close, 20)))"))
    assert t is None and "degenerate" in reason


def test_mul_same_subtree_rejected():
    # mul(x,x) = x² → 退化(方向单一、始终非负,非合格 alpha),拒
    t, reason = apply(parse("zscore(mul(ma(close, 20), ma(close, 20)))"))
    assert t is None and "degenerate" in reason


# ---------------- 过滤3:跨量纲 ----------------

def test_reject_cross_dimension():
    t, reason = apply(Node.elem("add", parse("ma(close, 20)"), parse("ma(volume, 20)")))
    # depth 2 → 先被 min_depth 拒;包裹一层截面到 depth 3 测跨量纲
    t2, reason2 = apply(parse("zscore(add(ma(close, 20), ma(volume, 20)))"))
    assert t2 is None and "cross_dimension" in reason2


def test_same_dimension_allowed():
    # overnight 与 intraday 同属 dimless → 可组合
    t, reason = apply(parse("zscore(add(ma(overnight, 20), ma(intraday, 20)))"))
    assert t is not None, reason


def test_price_and_mv_rejected():
    # close(price) 与 mv(mv) 不同量纲 → 拒
    t, reason = apply(parse("zscore(add(ma(close, 20), ma(mv, 20)))"))
    assert t is None and "cross_dimension" in reason


# ---------------- 综合 ----------------

def test_returns_simplified_tree():
    # 输入有冗余外层截面,通过审查后返回折叠后的树
    t, _ = apply(parse("zscore(zscore(sub(ma(close, 20), ma(low, 20))))"))
    assert t is not None
    assert t.to_str() == "zscore(sub(ma(close, 20), ma(low, 20)))"


# ---------------- 过滤1b:add/mul 交换结合规范化(2026-08-17)----------------

def test_simplify_flattens_and_orders_add():
    """嵌套 add 展平 + 字典序规范:等价排列得到同一写法/同一 hash。"""
    a = simplify(parse("add(rank_cs(log_mv), add(rank_cs(log_amount), zscore(ret)))"))
    b = simplify(parse("add(zscore(ret), add(rank_cs(log_amount), rank_cs(log_mv)))"))
    c = simplify(parse("add(add(rank_cs(log_amount), zscore(ret)), rank_cs(log_mv))"))
    assert a.to_str() == b.to_str() == c.to_str()
    assert a.expr_hash() == b.expr_hash() == c.expr_hash()


def test_simplify_flattens_mul_not_sub():
    """mul 同样规范化;sub 不交换,不得展平。"""
    a = simplify(parse("mul(rank_cs(ret), mul(zscore(ret), rank_cs(log_mv)))"))
    b = simplify(parse("mul(zscore(ret), mul(rank_cs(ret), rank_cs(log_mv)))"))
    assert a.to_str() == b.to_str()
    s = simplify(parse("sub(rank_cs(ret), sub(zscore(ret), rank_cs(log_mv)))"))
    assert s.to_str() == "sub(rank_cs(ret), sub(zscore(ret), rank_cs(log_mv)))"


def test_simplify_preserves_values():
    """规范化只改写法不改数值:展平前后面板求值一致。"""
    import numpy as np
    import pandas as pd
    from engine.expression import evaluate
    idx = pd.date_range("2020-01-01", periods=6)
    rng = np.random.default_rng(0)
    panels = {f: pd.DataFrame(rng.normal(0, 1, (6, 3)), index=idx)
              for f in ("ret", "log_mv", "log_amount")}
    e1 = evaluate(simplify(parse("add(rank_cs(log_mv), add(rank_cs(log_amount), zscore(ret)))")), panels)
    e2 = evaluate(parse("add(add(rank_cs(log_mv), rank_cs(log_amount)), zscore(ret))"), panels)
    assert np.allclose(e1.to_numpy(), e2.to_numpy(), equal_nan=True)


# ---------------- 过滤5:过度平滑/极值嵌套(2026-08-17,研报 §16)----------------

def test_reject_oversmoothed_std_std():
    t, reason = apply(parse("std(std(zscore(log_amount), 5), 20)"))
    assert t is None and "oversmoothed" in reason


def test_reject_oversmoothed_ma_std():
    # 跨算子的平滑嵌套(ma∘std)同样拒:统计量堆叠不分算子名
    t, reason = apply(parse("ma(std(zscore(ret), 10), 40)"))
    assert t is None and "oversmoothed" in reason


def test_reject_extreme_nesting():
    # 库内 6 因子的右半树形态:max(min(·,120),5) 极值嵌套 → 拒
    t, reason = apply(parse("zscore(max(min(down_shadow, 120), 5))"))
    assert t is None and "extreme_nesting" in reason


def test_reject_extreme_nesting_same_op():
    t, reason = apply(parse("zscore(min(min(down_shadow, 120), 5))"))
    assert t is None and "extreme_nesting" in reason


def test_single_smoothing_and_gap_nesting_pass():
    # 单层平滑 + 中间隔截面算子的嵌套(std(rank_cs(·)) 不是平滑嵌平滑)→ 通过
    t, reason = apply(parse("add(rank_cs(log_mv), std(zscore(log_amount), 20))"))
    assert t is not None, reason
    # 极值套平滑也不算「极值嵌极值」(如 max(ma(·),40) 是 winsorize 型组合)
    t2, reason2 = apply(parse("zscore(sub(max(ma(close, 5), 40), ma(close, 20)))"))
    assert t2 is not None, reason2


# ---------------- roc 语义闸(2026-08-18)----------------

def test_reject_roc_on_cs():
    """roc 直接作用于截面算子(rank_cs 有 [0,1] 下界、zscore 过零)→ 分母不良定义,拒。"""
    t, reason = apply(parse("add(rank_cs(log_mv), roc(rank_cs(log_mv), 60))"))
    assert t is None and "roc_on_cs" in reason
    t2, reason2 = apply(parse("zscore(roc(zscore(adj_close), 20))"))
    assert t2 is None and "roc_on_cs" in reason2


def test_roc_on_levels_and_delta_on_rank_pass():
    """roc 作用于价格水平(输出 dimless,与同量纲项组合)→ 合法;delta 作用于 rank → 合法。"""
    t, reason = apply(parse("add(roc(ma(close, 5), 10), ret)"))
    assert t is not None, reason
    t2, reason2 = apply(parse("add(rank_cs(log_mv), delta(rank_cs(log_mv), 5))"))
    assert t2 is not None, reason2

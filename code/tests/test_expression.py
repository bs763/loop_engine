# -*- coding: utf-8 -*-
"""expression.py 单元测试:解析/序列化往返、深度、字段、校验、求值、随机生成。"""
import numpy as np
import pandas as pd
import pytest

from engine.expression import Node, parse, evaluate, random_tree, expr_hash, MAX_DEPTH


# ---------------- 解析 / 序列化往返 ----------------

@pytest.mark.parametrize("s", [
    "close",
    "ma(close, 20)",
    "sub(high, low)",
    "zscore(close)",
    "zscore(div(ma(close, 20), std(sub(high, low), 10)))",
    "rank_cs(log_amount)",
    "div(ma(overnight, 118), min(down_shadow, 66))",
])
def test_parse_roundtrip(s):
    assert parse(s).to_str() == s


def test_parse_leaf():
    n = parse("close")
    assert n.is_leaf() and n.field == "close"


def test_parse_ts_window_and_child():
    n = parse("ma(close, 20)")
    assert n.op == "ma" and n.window == 20
    assert n.children[0].is_leaf() and n.children[0].field == "close"


def test_parse_elem_two_children():
    n = parse("sub(high, low)")
    assert n.op == "sub" and len(n.children) == 2
    assert n.children[0].field == "high" and n.children[1].field == "low"


def test_parse_unknown_op_raises():
    with pytest.raises(ValueError):
        parse("foobar(close)")


# ---------------- 深度 / 字段 ----------------

def test_depth():
    assert parse("close").depth() == 0
    assert parse("ma(close, 20)").depth() == 1
    assert parse("zscore(close)").depth() == 1
    assert parse("zscore(div(ma(close,20), std(sub(high,low),10)))").depth() == 4


def test_fields():
    n = parse("div(ma(close,20), std(sub(high,low),10))")
    assert n.fields() == {"close", "high", "low"}


# ---------------- 校验 ----------------

def test_validate_ok():
    parse("zscore(div(ma(close,20), std(sub(high,low),10)))").validate()


def test_validate_window_out_of_range():
    n = Node.ts("ma", Node.leaf("close"), 2)  # ma 最小 3
    with pytest.raises(ValueError):
        n.validate()


def test_validate_wrong_arity():
    n = Node(op="add", children=[Node.leaf("close")])  # elem 缺第二个子节点
    with pytest.raises(ValueError):
        n.validate()


# ---------------- 求值 ----------------

def _panels():
    return {
        "close": pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [10.0, 20, 30, 40]}),
        "high":  pd.DataFrame({"A": [2.0, 3, 4, 5], "B": [11.0, 21, 31, 41]}),
        "low":   pd.DataFrame({"A": [0.5, 1, 2, 3], "B": [9.0, 19, 29, 39]}),
    }


def test_evaluate_elem():
    r = evaluate(parse("sub(high, low)"), _panels())
    np.testing.assert_allclose(r["A"].values, [1.5, 2, 2, 2])


def test_evaluate_ts_then_cs():
    r = evaluate(parse("zscore(ma(close, 2))"), _panels())
    # ma(close,2) 第 1 行起有值;zscore 后逐行均值≈0、std≈1
    valid = r.dropna()
    np.testing.assert_allclose(valid.mean(axis=1).values, [0, 0, 0], atol=1e-12)


def test_evaluate_missing_field():
    with pytest.raises(KeyError):
        evaluate(parse("ma(unknown_xyz, 5)"), _panels())


# ---------------- 随机生成 ----------------

def test_random_tree_valid_and_bounded():
    rng = np.random.default_rng(42)
    fields = ["close", "high", "low", "volume", "overnight"]
    for _ in range(200):
        t = random_tree(fields, max_depth=MAX_DEPTH, rng=rng)
        t.validate()                       # 结构合法
        assert t.depth() <= MAX_DEPTH      # 深度有界
        assert t.fields().issubset(set(fields))  # 仅用给定字段


def test_random_tree_reproducible():
    fields = ["close", "volume"]
    t1 = random_tree(fields, rng=np.random.default_rng(7))
    t2 = random_tree(fields, rng=np.random.default_rng(7))
    assert t1.to_str() == t2.to_str()


def test_random_tree_not_all_leaf():
    # 多数应至少有 1 层算子(_depth<1 强制算子)
    rng = np.random.default_rng(0)
    fields = ["close", "high"]
    depths = [random_tree(fields, rng=rng).depth() for _ in range(100)]
    assert max(depths) >= 1


def test_random_tree_windows_on_grid():
    # 窗口只在 WINDOW_SET 规整值上(用户要求:不用 3/34/63 这种任意整数)
    from engine.config import WINDOW_SET
    rng = np.random.default_rng(42)
    fields = ["close", "high", "low", "overnight"]
    for _ in range(400):
        t = random_tree(fields, rng=rng)
        for node in t.walk():
            if node.window is not None:
                assert node.window in WINDOW_SET, f"窗口 {node.window} 不在 grid {WINDOW_SET}"


# ---------------- 哈希 ----------------

def test_expr_hash_stable_and_distinct():
    a = parse("ma(close, 20)")
    assert expr_hash(a) == expr_hash(parse("ma(close, 20)"))
    assert expr_hash(a) != expr_hash(parse("ma(close, 21)"))

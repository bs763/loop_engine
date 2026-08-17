# -*- coding: utf-8 -*-
"""operators.py 单元测试:14 算子数值正确性 + 注册表完整性。"""
import numpy as np
import pandas as pd
import pytest

from engine import operators as op


def panel(cols=("A", "B"), rows=None):
    """rows: dict col→list。返回宽表(RangeIndex × cols)。"""
    return pd.DataFrame(rows, columns=list(cols))


# ---------------- 注册表完整性 ----------------

def test_registry_counts():
    assert op.NUM_OPS == 14
    assert len(op.TS_OP_NAMES) == 8
    assert len(op.ELEM_OP_NAMES) == 4
    assert len(op.CS_OP_NAMES) == 2
    assert len(op.ALL_OP_NAMES) == 14


def test_registry_arity():
    for n in op.TS_OP_NAMES:
        assert op.OP_REGISTRY[n]["arity"] == 1
        assert op.OP_REGISTRY[n]["kind"] == "ts"
        assert op.OP_REGISTRY[n]["window_range"] is not None
    for n in op.ELEM_OP_NAMES:
        assert op.OP_REGISTRY[n]["arity"] == 2
        assert op.OP_REGISTRY[n]["kind"] == "elem"
    for n in op.CS_OP_NAMES:
        assert op.OP_REGISTRY[n]["arity"] == 1
        assert op.OP_REGISTRY[n]["kind"] == "cs"


def test_window_ranges():
    assert op.OP_REGISTRY["ma"]["window_range"] == (3, 250)
    assert op.OP_REGISTRY["skew"]["window_range"] == (10, 120)
    assert op.OP_REGISTRY["roc"]["window_range"] == (3, 60)


# ---------------- 时序算子 ----------------

def test_ma():
    p = panel(rows={"A": [1.0, 2, 3, 4, 5], "B": [10.0, 20, 30, 40, 50]})
    r = op.op_ma(p, 3)
    # 前 2 行 NaN(预热),之后 [2,3,4]
    assert r["A"].iloc[:2].isna().all()
    np.testing.assert_allclose(r["A"].iloc[2:].values, [2, 3, 4])
    np.testing.assert_allclose(r["B"].iloc[2:].values, [20, 30, 40])


def test_std_ddof1():
    p = panel(rows={"A": [1.0, 2, 3, 4, 5]})
    r = op.op_std(p, 3)
    # [1,2,3] 样本std(ddof=1)=1
    np.testing.assert_allclose(r["A"].iloc[2], 1.0)


def test_roc_and_delta():
    p = panel(rows={"A": [1.0, 2, 3, 4]})
    np.testing.assert_allclose(op.op_roc(p, 1)["A"].values, [np.nan, 1.0, 0.5, 1 / 3])
    np.testing.assert_allclose(op.op_delta(p, 2)["A"].values, [np.nan, np.nan, 2.0, 2.0])


def test_rank_ts_extremes():
    asc = panel(rows={"A": [1.0, 2, 3, 4, 5]})
    desc = panel(rows={"A": [5.0, 4, 3, 2, 1]})
    r_asc = op.op_rank_ts(asc, 3)
    r_desc = op.op_rank_ts(desc, 3)
    # 升序序列:当前值恒为窗口最大 → 1.0;降序 → 0.0
    np.testing.assert_allclose(r_asc["A"].iloc[2:].values, [1, 1, 1])
    np.testing.assert_allclose(r_desc["A"].iloc[2:].values, [0, 0, 0])


def test_max_min():
    p = panel(rows={"A": [3.0, 1, 4, 1, 5]})
    np.testing.assert_allclose(op.op_max(p, 3)["A"].iloc[2:].values, [4, 4, 5])
    np.testing.assert_allclose(op.op_min(p, 3)["A"].iloc[2:].values, [1, 1, 1])


# ---------------- 逐元素算子 ----------------

def test_elem_arith():
    a = panel(rows={"A": [1.0, 2], "B": [10.0, 20]})
    b = panel(rows={"A": [2.0, 2], "B": [5.0, 4]})
    np.testing.assert_allclose(op.op_add(a, b).values, [[3, 15], [4, 24]])
    np.testing.assert_allclose(op.op_sub(a, b).values, [[-1, 5], [0, 16]])
    np.testing.assert_allclose(op.op_mul(a, b).values, [[2, 50], [4, 80]])


def test_div_zero_is_nan_not_inf():
    a = panel(rows={"A": [1.0, 2.0]})
    b = panel(rows={"A": [0.0, 4.0]})
    r = op.op_div(a, b)
    assert np.isnan(r["A"].iloc[0])      # 除零 → NaN(非 inf)
    np.testing.assert_allclose(r["A"].iloc[1], 0.5)


def test_elem_aligns_mismatched_columns():
    a = panel(("A", "B"), {"A": [1.0, 2], "B": [3.0, 4]})
    c = panel(("B", "C"), {"B": [1.0, 1], "C": [10.0, 10]})
    r = op.op_add(a, c)
    # 仅 B 列两边都有 → 有值;A、C 仅一边 → NaN
    assert "A" in r and "C" in r
    np.testing.assert_allclose(r["B"].values, [4.0, 5.0])
    assert r["A"].isna().all() and r["C"].isna().all()


# ---------------- 截面算子 ----------------

def test_zscore_row_standardized():
    p = panel(("A", "B", "C"), {"A": [1.0, 10], "B": [2.0, 20], "C": [3.0, 30]})
    r = op.op_zscore(p)
    row_mean = r.mean(axis=1)
    row_std = r.std(axis=1, ddof=1)
    np.testing.assert_allclose(row_mean.values, [0, 0], atol=1e-12)
    np.testing.assert_allclose(row_std.values, [1, 1], atol=1e-12)


def test_rank_cs_in_unit_interval():
    p = panel(("A", "B", "C"), {"A": [3.0, 1], "B": [1.0, 2], "C": [2.0, 3]})
    r = op.op_rank_cs(p)
    assert ((r >= 0) & (r <= 1)).all().all()
    # 第一行:A=3 最大→1.0,B=1 最小→1/3
    np.testing.assert_allclose(r.iloc[0].values, [1.0, 1 / 3, 2 / 3])


# ---------------- 分发 ----------------

def test_apply_dispatch():
    p = panel(rows={"A": [1.0, 2, 3, 4, 5]})
    via_apply = op.apply("ma", [p], window=3)
    direct = op.op_ma(p, 3)
    pd.testing.assert_frame_equal(via_apply, direct)
    with pytest.raises(ValueError):
        op.apply("ma", [p])  # 缺 window
    with pytest.raises(KeyError):
        op.apply("nope", [p])


def test_field_dimensions():
    assert op.field_dimension("close") == op.DIM_PRICE
    assert op.field_dimension("volume") == op.DIM_VOLUME
    assert op.field_dimension("overnight") == op.DIM_DIMLESS
    assert op.field_dimension("mv") == op.DIM_MV
    assert op.field_dimension("unknown_xyz") == op.DIM_DIMLESS  # 未知→保守 dimless


def test_roc_sanitizes_inf():
    """op_roc 除零 ±inf 必须转 NaN(2019-04-18 事故根因:inf 毒化截面 zscore)。"""
    import numpy as np
    import pandas as pd
    from engine.operators import op_roc
    p = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0], "B": [1.0, 2.0, 4.0, 8.0]})
    r = op_roc(p, 1)
    assert np.isinf(r.to_numpy()).sum() == 0
    assert np.isnan(r.loc[1, "A"])   # 1/0 → inf → NaN


def test_zscore_robust_to_single_inf():
    """单个 inf 毒化整截面的事故防线:zscore 入口消毒,坏列自身 NaN、其余列正常标准化。"""
    import numpy as np
    import pandas as pd
    from engine.operators import op_zscore
    idx = pd.date_range("2020-01-01", periods=3)
    p = pd.DataFrame({"A": [np.inf, 1.0, -1.0], "B": [2.0, 4.0, 6.0], "C": [1.0, 2.0, 3.0]}, index=idx)
    z = op_zscore(p)
    assert np.isnan(z.loc[idx[0], "A"])          # inf → NaN(不再毒化整行)
    assert not z.loc[idx[0], ["B", "C"]].isna().any()   # 同日其它股票正常出值
    assert np.isinf(z.to_numpy()).sum() == 0


def test_leaf_panels_sanitized():
    """求值入口叶子消毒:原始面板混入的 ±inf 不进管线。"""
    import numpy as np
    import pandas as pd
    from engine.expression import Node, evaluate
    idx = pd.date_range("2020-01-01", periods=3)
    panels = {"ret": pd.DataFrame([[np.inf, 0.01], [-np.inf, 0.02], [0.0, 0.03]], index=idx,
                                  columns=["A", "B"])}
    out = evaluate(Node.leaf("ret"), panels)
    assert np.isinf(out.to_numpy()).sum() == 0

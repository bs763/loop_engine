# -*- coding: utf-8 -*-
"""filters.py 测试:用构造的 FactorMetrics 验证 11 项过滤各命中/全过。"""
import numpy as np

from backtest.interface import FactorMetrics
from filters import apply_filters


def good_metrics(**over) -> FactorMetrics:
    rng = np.random.default_rng(0)
    nav = np.cumprod(1.0 + rng.normal(0.002, 0.005, size=1942))  # 强正漂移 → 末年夏普/滚动均稳定 >0
    base = dict(
        ic_mean=0.06, icir=0.8, icir_annual=5.0, t_stat_nw=10.0, positive_ratio=0.85,
        ls_annual=0.4, ls_sharpe=1.5, ls_max_dd=-0.1, calmar=2.0,
        long_excess_annual=0.1, long_excess_sharpe=1.2, monotonicity=0.95,
        annual_ls_return={y: 0.3 for y in range(2018, 2026)},
        annual_ic={y: 0.05 for y in range(2018, 2026)},
        ic_series=rng.normal(0.06, 0.08, size=1942).tolist(),
        long_excess_nav=nav.tolist(), ls_nav=nav.tolist(),
    )
    base.update(over)
    return FactorMetrics(**base)


def test_pass_all():
    r = apply_filters(good_metrics())
    assert r.passed, r.reasons


def test_fail_ic_gate():
    r = apply_filters(good_metrics(ic_mean=0.02))
    assert not r.passed and any("|IC|" in x for x in r.reasons)


def test_fail_icir():
    r = apply_filters(good_metrics(icir=0.2))   # ICIR 不足
    assert not r.passed and any("ICIR" in x for x in r.reasons)


def test_fail_year_negative():
    m = good_metrics()
    m.annual_ls_return[2020] = -0.1
    r = apply_filters(m)
    assert not r.passed and any("多空≤0的年份" in x for x in r.reasons)


def test_fail_year_missing():
    m = good_metrics()
    del m.annual_ls_return[2025]
    r = apply_filters(m)
    assert not r.passed and any("缺年份" in x for x in r.reasons)


def test_fail_sharpe():
    r = apply_filters(good_metrics(ls_sharpe=0.3))
    assert not r.passed and any("整体夏普" in x for x in r.reasons)


def test_fail_calmar():
    r = apply_filters(good_metrics(calmar=0.5))
    assert not r.passed and any("Calmar" in x for x in r.reasons)


def test_fail_long_excess():
    r = apply_filters(good_metrics(long_excess_annual=-0.05))   # 多头超额为负
    assert not r.passed and any("多头超额年化" in x for x in r.reasons)


def test_fail_monotonicity():
    r = apply_filters(good_metrics(monotonicity=0.80))          # 单调性不足
    assert not r.passed and any("单调性" in x for x in r.reasons)


def test_fail_rolling():
    # 前 1642 日上升、末 300 日下降 → 近 12 月多空 < 0、末年夏普 < 0
    nav = list(np.concatenate([np.linspace(1, 2, 1642), np.linspace(2, 1, 300)]))
    r = apply_filters(good_metrics(ls_nav=nav))
    assert not r.passed
    assert any(("近12月" in x or "近9月" in x or "末年夏普" in x) for x in r.reasons)


def test_fail_corr():
    m = good_metrics()
    r = apply_filters(m, stored_ic_series=[m.ic_series])  # 与自身完全相关
    assert not r.passed and any("IC相关性" in x for x in r.reasons)


def test_corr_below_threshold_ok():
    rng = np.random.default_rng(1)
    other = rng.normal(0, 0.08, size=1942).tolist()  # 几乎不相关
    r = apply_filters(good_metrics(), stored_ic_series=[other])
    assert r.passed, r.reasons


def test_corr_replace_when_better():
    m = good_metrics()  # ic_mean=0.06, icir=0.8, monotonicity=0.95, long_excess=0.1
    old = {"hash": "old", "ic_series": m.ic_series,
           "metrics": {"ic_mean": 0.04, "icir": 0.5, "monotonicity": 0.90, "long_excess_annual": 0.03}}
    r = apply_filters(m, stored_factors=[old])
    assert r.passed and r.replace_hashes == ["old"]     # 高相关但更优 → 替换信号


def test_corr_reject_when_worse():
    m = good_metrics()
    old = {"hash": "old", "ic_series": m.ic_series,
           "metrics": {"ic_mean": 0.08, "icir": 1.0, "monotonicity": 0.98, "long_excess_annual": 0.15}}
    r = apply_filters(m, stored_factors=[old])
    assert not r.passed and any("IC相关性" in x for x in r.reasons)   # 高相关但更差 → 拒


def test_fail_fsa():
    from engine.expression import parse
    from engine.fsa import FSA
    fsa = FSA()
    sk = "zscore(sub(ma(FLD, N), ma(FLD, N)))"
    for _ in range(3):       # 占比 3/4=75% > 15% 且 ≥2 → 冻结
        fsa.observe(sk)
    fsa.observe("other(FLD)")
    node = parse("zscore(sub(ma(close, 20), ma(low, 20)))")
    r = apply_filters(good_metrics(), fsa=fsa, node=node)
    assert not r.passed and any("10." in x for x in r.reasons)


def test_fail_mode_lib():
    r = apply_filters(good_metrics(), failed_hashes={"abc"}, expr_hash="abc")
    assert not r.passed and any("失败模式库" in x for x in r.reasons)


def test_fail_family_subtree_cap():
    # 右半树骨架 zscore(max(min(FLD,N),N)) 已在 2 个库存因子,且库存质量更高 → 第 3 个拒
    from engine.expression import parse
    strong = {"ic_mean": 0.08, "icir": 1.0, "monotonicity": 1.0, "long_excess_annual": 0.15}
    stored = [
        {"expr": "add(ma(log_mv, 20), zscore(max(min(up_shadow, 120), 5)))", "hash": "a", "metrics": strong},
        {"expr": "add(std(log_amount, 40), zscore(max(min(down_shadow, 120), 5)))", "hash": "b", "metrics": strong},
    ]
    node = parse("add(rank_cs(ret), zscore(max(min(hl_ratio, 120), 5)))")
    r = apply_filters(good_metrics(), stored_factors=stored, node=node)
    assert not r.passed and any("同构子树" in x for x in r.reasons)
    assert not r.replace_hashes       # 质量未全面更优 → 不替换


def test_family_replace_when_better():
    # 同构家族超限但候选质量全面更优(×1.05)→ 保优淘劣:替换该族全部旧因子入库
    from engine.expression import parse
    weak = {"ic_mean": 0.04, "icir": 0.5, "monotonicity": 0.9, "long_excess_annual": 0.03}
    stored = [
        {"expr": "add(ma(log_mv, 20), zscore(max(min(up_shadow, 120), 5)))", "hash": "a", "metrics": weak},
        {"expr": "add(std(log_amount, 40), zscore(max(min(down_shadow, 120), 5)))", "hash": "b", "metrics": weak},
    ]
    node = parse("add(rank_cs(ret), zscore(max(min(hl_ratio, 120), 5)))")
    r = apply_filters(good_metrics(), stored_factors=stored, node=node)
    assert r.passed, r.reasons
    assert r.replace_hashes == ["a", "b"]   # 替换信号:两个旧因子都让位


def test_family_below_cap_ok():
    # 同构子树只在 1 个库存因子出现(< 上限 2)→ 放行
    from engine.expression import parse
    stored = [{"expr": "add(ma(log_mv, 20), zscore(max(min(up_shadow, 120), 5)))"}]
    node = parse("add(rank_cs(ret), zscore(max(min(hl_ratio, 120), 5)))")
    r = apply_filters(good_metrics(), stored_factors=stored, node=node)
    assert r.passed, r.reasons


def test_family_generic_small_subtrees_ignored():
    # 3 节点通用件(如 std(zscore(FLD),N))不参与家族计数 → 多个库存含它也不拒
    from engine.expression import parse
    stored = [
        {"expr": "add(std(zscore(log_amount), 20), rank_cs(ret))"},
        {"expr": "mul(std(zscore(log_amount), 40), rank_cs(overnight))"},
    ]
    node = parse("sub(std(zscore(log_amount), 10), rank_cs(intraday))")
    r = apply_filters(good_metrics(), stored_factors=stored, node=node)
    assert r.passed, r.reasons

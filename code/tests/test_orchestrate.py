# -*- coding: utf-8 -*-
"""loop_orchestrate.py 测试:mock 单轮跑通、检查点写/恢复、入库与 FSA 累积。"""
import numpy as np
import pandas as pd

from backtest.interface import Evaluator, FactorMetrics
from backtest.mock import MockEvaluator
from engine.checkpoint import Checkpoint
from engine.evolve import Evolver
from engine.fsa import FSA
from loop_orchestrate import run_round, build_field_panels, restore_fsa

FIELDS = ["adj_close", "overnight", "amplitude", "log_volume", "ret"]


def _synth_panels(n_days=60, n_stocks=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n_days)
    stocks = [f"S{i:03d}.XSHG" for i in range(n_stocks)]
    return {f: pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=stocks)
            for f in FIELDS}


class AlwaysPass(Evaluator):
    """总返回过十一项过滤的指标(用于演练入库/FSA/去重路径)。"""
    def evaluate(self, panel, name="factor"):
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        nav = np.cumprod(1.0 + rng.normal(0.002, 0.005, size=1942))
        return FactorMetrics(
            ic_mean=0.06, icir=0.8, icir_annual=5.0, t_stat_nw=10.0, positive_ratio=0.85,
            ls_annual=0.4, ls_sharpe=1.5, ls_max_dd=-0.1, calmar=2.0,
            long_excess_annual=0.1, long_excess_sharpe=1.2, monotonicity=0.95, direction=1,
            annual_ls_return={y: 0.3 for y in range(2018, 2026)},
            annual_ic={y: 0.05 for y in range(2018, 2026)},
            ic_series=rng.normal(0.06, 0.08, size=1942).tolist(),
            long_excess_nav=nav.tolist(), ls_nav=nav.tolist(), meta={"name": name},
        )


def test_run_round_mock_restores(tmp_path):
    panels = _synth_panels()
    cp = Checkpoint(tmp_path / "cp.json")
    ev = Evolver(FIELDS, rng=np.random.default_rng(1))
    stats = run_round(checkpoint=cp, evolver=ev, evaluator=MockEvaluator(seed=2),
                      field_panels=panels, fsa=FSA(), fields=FIELDS, n_candidates=30)
    assert stats.n_generated == 30 and stats.iteration == 1
    # 检查点恢复一致
    cp2 = Checkpoint.load(tmp_path / "cp.json")
    assert cp2.iteration == 1
    assert len(cp2.tested_hashes) == stats.n_tested
    assert len(cp2.stored_factors) == stats.n_pass_filters


def test_coverage_reason_detects_collapse():
    """覆盖率防线:单月塌陷 < 本地中位×0.6 → 报原因;正常面板 → None。"""
    from loop_orchestrate import _coverage_reason
    idx = pd.date_range("2018-01-01", periods=900, freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (900, 5)), index=idx)
    assert _coverage_reason(df) is None                      # 正常
    hole = (idx >= "2019-05-01") & (idx <= "2019-05-31")     # 单月全 NaN
    df.loc[hole, :] = np.nan
    r = _coverage_reason(df)
    assert r is not None and r.startswith("ValueError") and "2019-05" in r


def test_coverage_ignores_warmup_months():
    """warmup 期(回测窗口前的滚动窗预热,天然 0% 覆盖)不得触发塌陷——
    全库体检曾暴露此误杀,会导致所有候选被拒。"""
    from loop_orchestrate import _coverage_reason
    idx = pd.date_range("2017-01-01", periods=1100, freq="B")   # 2017 为 warmup
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 1, (1100, 5)), index=idx)
    df.loc[:"2017-12-31"] = np.nan                               # warmup 全 NaN
    assert _coverage_reason(df) is None                          # 不得误报


def test_simplify_or_combine():
    """分支支配简化(用户 2026-08-18):std 比 ≥3x 取支配支;平衡则合成面板(等价于 evaluate)。"""
    from engine.expression import parse
    from loop_orchestrate import _simplify_or_combine
    idx = pd.date_range("2020-01-01", periods=60)
    rng = np.random.default_rng(3)
    small = pd.DataFrame(rng.uniform(0, 1, (60, 4)), index=idx)      # std≈0.29
    big = pd.DataFrame(rng.normal(0, 5, (60, 4)), index=idx)         # std≈5
    node = parse("add(a, b)")                                        # 占位算子名无关紧要
    n2, panel = _simplify_or_combine(node, big, small)
    assert n2 is node.children[0] and panel is big                  # 支配支胜出
    n3, panel3 = _simplify_or_combine(node, small, small.copy())
    assert n3 is node and np.allclose(panel3.values, small.values * 2, equal_nan=True)


def test_build_field_panels():
    df = pd.DataFrame({
        "order_book_id": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2018-01-02", "2018-01-03"] * 2),
        "ret": [0.1, 0.2, 0.3, 0.4],
    })
    panels = build_field_panels(df, ["ret"])
    assert panels["ret"].shape == (2, 2)
    assert list(panels["ret"].columns) == ["A", "B"]


def test_llm_final_veto_blocks_store(tmp_path):
    """LLM 终审(2026-08-17 接线):全过滤通过后终审拒 → 不入库;终审放行 → 正常入库。"""
    from llm.provider import MockProvider
    panels = _synth_panels()
    # 拒绝版
    cp = Checkpoint(tmp_path / "cp_veto.json")
    s = run_round(checkpoint=cp, evolver=Evolver(FIELDS, rng=np.random.default_rng(5)),
                  evaluator=AlwaysPass(), field_panels=panels, fsa=FSA(), fields=FIELDS,
                  n_candidates=40, llm_reviewer=MockProvider(responder=lambda p: "REJECT: 测试否决"))
    assert s.n_pass_filters == 0 and len(cp.stored_factors) == 0
    # 放行版
    cp2 = Checkpoint(tmp_path / "cp_ok.json")
    s2 = run_round(checkpoint=cp2, evolver=Evolver(FIELDS, rng=np.random.default_rng(5)),
                   evaluator=AlwaysPass(), field_panels=panels, fsa=FSA(), fields=FIELDS,
                   n_candidates=40, llm_reviewer=MockProvider(responder=lambda p: "ACCEPT: ok"))
    assert s2.n_pass_filters > 0 and len(cp2.stored_factors) == s2.n_pass_filters


def test_store_accumulates_and_fsa_persists(tmp_path):
    panels = _synth_panels()
    cp = Checkpoint(tmp_path / "cp.json")
    fsa = FSA()
    ev = Evolver(FIELDS, rng=np.random.default_rng(3))
    # 第 1 轮
    s1 = run_round(checkpoint=cp, evolver=ev, evaluator=AlwaysPass(),
                   field_panels=panels, fsa=fsa, fields=FIELDS, n_candidates=40)
    assert s1.n_pass_filters > 0, "AlwaysPass 下应有审查通过的因子入库"
    assert s1.stored_total == s1.n_pass_filters

    # 第 2 轮:从恢复的检查点 + FSA 继续(模拟断点续跑)
    cp2 = Checkpoint.load(tmp_path / "cp.json")
    fsa2 = restore_fsa(cp2)
    assert sum(fsa2.counts.values()) == s1.n_pass_filters  # FSA 计数已持久化
    ev2 = Evolver(FIELDS, rng=np.random.default_rng(4))
    s2 = run_round(checkpoint=cp2, evolver=ev2, evaluator=AlwaysPass(),
                   field_panels=panels, fsa=fsa2, fields=FIELDS, n_candidates=40)
    assert s2.iteration == 2
    assert cp2.stored_factors[0]["expr"]  # 入库记录有表达式

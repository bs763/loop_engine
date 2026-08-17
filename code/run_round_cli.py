# -*- coding: utf-8 -*-
"""单轮因子挖掘 CLI —— 供 OS cron 或 Claude `/loop` 每次调用。

  mock 模式(离线,无 LLM / 无 alphalab):
      uv run --directory factor_loop_engine code/run_round_cli.py --mock --n 100
  真实模式(GLM-5.3 生成 + LLM 终审 + alphalab 回测,2018-2025):
      uv run --directory factor_loop_engine code/run_round_cli.py --n 100

每轮:生成 N 候选 → 审查五过滤 → 回测 → 十一项过滤 → 入库 → 检查点原子落盘(断点续跑)。
真实模式首次会缓存字段宽表到 cache/panels/(之后秒级加载)。
最后一行 `STATUS: ...` 供 hook / 监控解析。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from adaptive import dynamic_budget, round_signals
from backtest.alphalab_adapter import AlphalabEvaluator
from backtest.mock import MockEvaluator
from engine.checkpoint import Checkpoint
from engine.evolve import Evolver
from engine.fsa import FSA
from llm.mechanisms import make_evolve_llm_hook
from llm.settings import generation_provider, review_provider
from loop_orchestrate import build_field_panels, restore_fsa, run_round
from paths import CACHE_DIR, OUTPUT_DIR, PROJECT_ROOT

# 引擎可用字段(均有 FIELD_DIM 量纲,供 review 跨量纲过滤)
FIELDS = ["adj_close", "adj_high", "adj_low", "overnight", "intraday", "amplitude",
          "up_shadow", "down_shadow", "hl_ratio", "ret", "log_volume", "log_amount", "log_mv"]
PANELS_CACHE = CACHE_DIR / "panels"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "checkpoint.json"
DEFAULT_ALPHALAB_CONFIG = PROJECT_ROOT / "config" / "alphalab.yaml"   # 项目专用副本(h5-only、无 barra)


def _synth_panels() -> dict[str, pd.DataFrame]:
    """mock 模式用合成面板。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-02", periods=60)
    stocks = [f"S{i:03d}.XSHG" for i in range(20)]
    return {f: pd.DataFrame(rng.normal(0, 1, (60, 20)), index=dates, columns=stocks)
            for f in FIELDS}


def _real_panels() -> dict[str, pd.DataFrame]:
    """真实模式:载 COMPUTE_START_YEAR(2015)-2025 因子数据 → 字段宽表(含 warmup buffer)。

    缓存按起始年校验:若缓存面板起始年晚于 COMPUTE_START_YEAR(如旧的 2018 缓存),自动重建。
    """
    from engine.config import COMPUTE_START_YEAR
    PANELS_CACHE.mkdir(parents=True, exist_ok=True)
    if all((PANELS_CACHE / f"{f}.parquet").exists() for f in FIELDS):
        first_min_year = pd.read_parquet(PANELS_CACHE / f"{FIELDS[0]}.parquet").index.min().year
        if first_min_year <= COMPUTE_START_YEAR:
            return {f: pd.read_parquet(PANELS_CACHE / f"{f}.parquet") for f in FIELDS}
        print(f"缓存面板起始年={first_min_year} > {COMPUTE_START_YEAR}(warmup),重建…", flush=True)
    from data_layer import load_factor_data
    print(f"构建字段宽表({COMPUTE_START_YEAR}-2025,缓存到 cache/panels/)…", flush=True)
    df = load_factor_data(COMPUTE_START_YEAR, 2025)
    panels = build_field_panels(df, FIELDS)
    for f, p in panels.items():
        p.to_parquet(PANELS_CACHE / f"{f}.parquet")
    return panels


def _in_peak_hours(now: datetime | None = None) -> bool:
    """北京高峰时段(每天 9:00-12:00、14:00-18:00)是否应暂停 loop。"""
    t = now or datetime.now()
    return (9 <= t.hour < 12) or (14 <= t.hour < 18)


def main() -> None:
    ap = argparse.ArgumentParser(description="单轮因子挖掘(供 cron / /loop 调用)")
    ap.add_argument("--mock", action="store_true", help="mock 回测 + 无 LLM(离线)")
    ap.add_argument("--n", type=int, default=100, help="每轮候选数")
    ap.add_argument("--workers", type=int, default=3,
                    help="回测并行线程数(默认 3;基准实测 0.4.2 单次回测 ~37s,本机 6 物理核下 1→2→3 线性加速 2.63x、"
                         "3→4 无增益、6 会丢回测,故 3 为甜点。ThreadPool 在 alphalab 子进程等待时释放 GIL,"
                         "表达式求值仅 ~4s 占比小,不必上进程池)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--alphalab-config", default=str(DEFAULT_ALPHALAB_CONFIG),
                    help="alphalab yaml 配置(默认项目专用 config/alphalab.yaml)")
    ap.add_argument("--force", action="store_true",
                    help="忽略高峰时段暂停(默认 9-12/14-18 跳过;仅用户显式要求时用)")
    args = ap.parse_args()

    if _in_peak_hours() and not args.force:
        print("高峰时段(北京时间 9:00-12:00、14:00-18:00),跳过本轮。", flush=True)
        return

    cp = Checkpoint.load(args.checkpoint)            # 断点续跑
    cfg, breason = dynamic_budget(cp.history)        # 自适应预算(M3)
    print(f"BUDGET: {breason} | mutate/crossover/perturb/random/llm = "
          f"{cfg.mutate}/{cfg.crossover}/{cfg.perturb}/{cfg.random}/{cfg.llm}")

    gen_provider, rev_provider = None, None
    if args.mock:
        panels = _synth_panels()
        evaluator = MockEvaluator()
        evolver = Evolver(FIELDS, config=cfg, rng=np.random.default_rng())           # 无 llm_provider
    else:
        panels = _real_panels()
        evaluator = AlphalabEvaluator(horizon=5, config_yaml=args.alphalab_config)
        gen_provider = generation_provider()
        rev_provider = review_provider()
        evolver = Evolver(FIELDS, config=cfg, rng=np.random.default_rng(),
                          llm_provider=make_evolve_llm_hook(gen_provider))  # GLM 生成(机制引导)

    fsa = restore_fsa(cp) if cp.fsa_state else FSA()
    stats = run_round(checkpoint=cp, evolver=evolver, evaluator=evaluator,
                      field_panels=panels, fsa=fsa, fields=FIELDS,
                      n_candidates=args.n, n_workers=args.workers,
                      llm_reviewer=rev_provider)  # None(mock)→跳过 LLM 终审
    print(stats)
    if stats.sample_reject_reasons:
        print("REJECTS:")
        for r in stats.sample_reject_reasons:
            print(f"  - {r}")
    # 拒绝分类汇总(每轮窗口一句话)
    if stats.reject_summary:
        items = ", ".join(f"{k}×{v}" for k, v in
                          sorted(stats.reject_summary.items(), key=lambda kv: -kv[1]))
        print(f"REJECT分类: {items}")
    # 保优淘劣:高相关但综合质量更优 → 替换旧因子(单独的「replaced」记录)
    replaced = [r for r in stats.reject_records if r.get("disp") == "replaced"]
    if replaced:
        print(f"保优淘劣: {len(replaced)} 个高相关但更优 → 已替换旧因子")
        for r in replaced:
            print(f"  - {r['expr']} ({r['reasons'][0] if r['reasons'] else ''})")
    # 拒绝日志(每候选一行,追溯历史被检因子+理由)
    if stats.reject_records:
        with open(OUTPUT_DIR / "rejects.jsonl", "a", encoding="utf-8") as f:
            for r in stats.reject_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # LLM 健康度(mechanisms._bump 计数;静默兜底否则不可见)
    if gen_provider is not None:
        def _cnt(p, k):
            return getattr(p, k, 0) if p is not None else 0
        g = {k: _cnt(gen_provider, k) for k in ("llm_gen_ok", "llm_gen_bad_output",
                                                "llm_gen_api_error", "llm_gen_fallback")}
        r_ok, r_err = _cnt(rev_provider, "llm_rev_ok"), _cnt(rev_provider, "llm_rev_error")
        if any(g.values()) or r_ok or r_err:
            print(f"LLM: 生成 ok{g['llm_gen_ok']}/解析失败{g['llm_gen_bad_output']}/"
                  f"API错{g['llm_gen_api_error']}/兜底{g['llm_gen_fallback']}"
                  f" | 终审 ok{r_ok}/错{r_err}")
    print(f"STATUS: iter={stats.iteration} tested={len(cp.tested_hashes)} "
          f"stored={stats.stored_total} new={stats.n_pass_filters} "
          f"elapsed={stats.elapsed_sec/60:.1f}min workers={args.workers}")
    print(f"SIGNALS: {round_signals(cp)}")


if __name__ == "__main__":
    main()

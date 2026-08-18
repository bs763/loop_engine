# -*- coding: utf-8 -*-
"""把入库因子导出成 parquet 值表(date×股票,alphalab 同款格式)+ manifest 清单。

跑法:
  uv run --directory factor_loop_engine code/export_factors.py            # 真实数据(默认)
  uv run --directory factor_loop_engine code/export_factors.py --mock     # 合成面板(演示)
读 output/checkpoint.json 的入库因子 → 在 2018-2025 字段面板上求值 → 落 output/factors/。
导出的 parquet 可直接喂回 alphalab 或用于策略。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from engine.checkpoint import Checkpoint
from engine.expression import evaluate as eval_expr, parse
from paths import OUTPUT_DIR


def _slug(expr: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", expr)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="导出入库因子为 parquet 值表 + 清单")
    ap.add_argument("--checkpoint", default=str(OUTPUT_DIR / "checkpoint.json"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "factors"))
    ap.add_argument("--mock", action="store_true", help="用合成面板(演示;默认真实数据)")
    args = ap.parse_args()

    cp = Checkpoint.load(args.checkpoint)
    factors = cp.stored_factors
    if not factors:
        print(f"入库因子为 0({args.checkpoint})。先跑真实模式攒因子,再导出。")
        return

    from run_round_cli import FIELDS, _real_panels, _synth_panels
    print(f"入库 {len(factors)} 个因子,构建字段面板…")
    panels = _synth_panels() if args.mock else _real_panels()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for i, f in enumerate(factors, 1):
        expr = f.get("expr", "")
        try:
            panel = eval_expr(parse(expr), panels)
        except Exception as e:  # noqa: BLE001
            failed.append((expr, str(e)))
            continue
        df = panel.copy()
        df.index.name = "date"
        df.columns = df.columns.astype(str)
        df = df.astype("float32")
        from engine.config import BACKTEST_END, BACKTEST_START
        df = df.loc[BACKTEST_START:BACKTEST_END]   # 只导出回测窗口 2018-2025(已 warmup)
        h = (f.get("hash", "") or str(i))[:8]
        fname = f"{i:04d}_{_slug(expr)}_{h}.parquet"
        df.to_parquet(out / fname)
        m = f.get("metrics", {})
        om = f.get("oos_metrics") or {}          # OOS(2025)存档——只报告,不参与筛选
        rows.append({
            "idx": i, "file": fname, "expr": expr, "hash": f.get("hash", ""),
            "direction": m.get("direction"), "ic_mean": m.get("ic_mean"),
            "icir": m.get("icir"), "ls_annual": m.get("ls_annual"),
            "ls_sharpe": m.get("ls_sharpe"), "calmar": m.get("calmar"),
            "long_excess_annual": m.get("long_excess_annual"),
            "monotonicity": m.get("monotonicity"),
            "oos_ic_mean": om.get("ic_mean"), "oos_icir": om.get("icir"),
            "oos_ls_sharpe": om.get("ls_sharpe"),
            "oos_long_excess_annual": om.get("long_excess_annual"),
            "oos_monotonicity": om.get("monotonicity"),
        })

    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    # OOS 报告(样本外 2025):按 OOS IC 降序 + IC 保有率,单独成档
    if any(r.get("oos_ic_mean") is not None for r in rows):
        oos = pd.DataFrame([{k: r[k] for k in ("expr", "hash", "ic_mean", "oos_ic_mean",
                                               "ls_sharpe", "oos_ls_sharpe",
                                               "oos_long_excess_annual", "oos_monotonicity")} |
                            {"ic_retention": (r["oos_ic_mean"] / r["ic_mean"]) if r.get("ic_mean") else None}
                            for r in rows if r.get("oos_ic_mean") is not None])
        oos = oos.sort_values("oos_ic_mean", ascending=False)
        oos.to_csv(out / "oos_report.csv", index=False, encoding="utf-8-sig")
        print(f"OOS 报告 → {out / 'oos_report.csv'}({len(oos)} 个,含 IC 保有率)")
    print(f"导出 {len(rows)} 个因子 → {out}/")
    print(f"清单 → {out / 'manifest.csv'}")
    # 清理被替换/移除的旧因子残留 parquet(保优淘劣换 hash 后,旧文件不再对应任何入库因子)
    keep = {r["file"] for r in rows}
    stale = [p for p in out.glob("*.parquet") if p.name not in keep]
    for p in stale:
        p.unlink()
    if stale:
        print(f"清理 {len(stale)} 个旧残留 parquet")
    if failed:
        print(f"失败 {len(failed)} 个:")
        for e, _ in failed[:5]:
            print(f"  {e[:70]}")


if __name__ == "__main__":
    main()

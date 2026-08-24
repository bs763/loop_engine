# -*- coding: utf-8 -*-
"""一次性回填在库因子的 ls_ret(多空日收益,PnL 口径相关性观察用,用户 2026-08-24)。

历史因子入库时未持久化 ls_nav;本脚本对缺 ls_ret 的在库因子重跑本地 alphalab
回测(h5 缓存,无网络配额)提取 ls_nav → 差分存 checkpoint。跑前自动备份。

跑法:  uv run code/backfill_ls_ret.py
"""
from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, "code")

from backtest.alphalab_adapter import AlphalabEvaluator
from engine.checkpoint import Checkpoint
from engine.expression import evaluate, parse
from paths import CACHE_DIR, OUTPUT_DIR, PROJECT_ROOT

PANELS_CACHE = CACHE_DIR / "panels"
FIELDS = ["adj_close", "adj_high", "adj_low", "overnight", "intraday", "amplitude",
          "up_shadow", "down_shadow", "hl_ratio", "ret", "log_volume", "log_amount", "log_mv"]


def main() -> None:
    ckpt_path = OUTPUT_DIR / "checkpoint.json"
    cp = Checkpoint.load(str(ckpt_path))
    todo = [f for f in cp.stored_factors if not f.get("ls_ret") and f.get("expr")]
    if not todo:
        print("全部在库因子已有 ls_ret,无需回填。")
        return
    bak = ckpt_path.with_name(f"checkpoint.json.bak-lsret-{datetime.now():%Y%m%d}")
    shutil.copy(ckpt_path, bak)
    print(f"待回填 {len(todo)}/{len(cp.stored_factors)} 个,已备份 → {bak.name}")

    panels = {f: pd.read_parquet(PANELS_CACHE / f"{f}.parquet") for f in FIELDS}
    evaluator = AlphalabEvaluator(horizon=5,
                                  config_yaml=str(PROJECT_ROOT / "config" / "alphalab.yaml"))

    def _one(factor: dict) -> tuple[dict, list | None]:
        try:
            panel = evaluate(parse(factor["expr"]), panels)
            m = evaluator.evaluate(panel, name="bf_" + factor["hash"][:10])
            return factor, m.ls_nav
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {factor['expr'][:50]}: {type(e).__name__}: {str(e)[:120]}")
            return factor, None

    t0 = time.perf_counter()
    ok = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for factor, nav in ex.map(_one, todo):
            if nav is not None:
                factor["ls_ret"] = np.diff(np.asarray(nav, dtype=float)).tolist()
                ok += 1
    cp.save()
    print(f"回填完成: {ok}/{len(todo)} 个,耗时 {(time.perf_counter()-t0)/60:.1f}min")

    from lib_status import corr_report_lines
    print("\n=== 回填后双口径相关性体检 ===")
    for line in corr_report_lines(cp.stored_factors):
        print(line)


if __name__ == "__main__":
    main()

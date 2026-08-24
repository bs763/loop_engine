# -*- coding: utf-8 -*-
"""一次性回填失败模式库(用户 2026-08-24):从 output/rejects.jsonl 重建 output/failed_patterns.json。

口径与运行时记录一致(engine/failed_patterns.record_reject):
  - hash 去重(同表达式终身只算一次,与 tested_hashes 语义对齐);
  - filter_reject 拒因全为 #9/#15 → 占位灭(occupied),不算死证据;
  - review_reject / backtest_error → 内因灭(fails);
  - disp=stored/replaced → 成功史(stored),另与 checkpoint.stored_factors 合并。

跑法:  uv run code/rebuild_failed_patterns.py   (项目根下,覆盖已有 failed_patterns.json)
"""
from __future__ import annotations

import json
import sys

from paths import OUTPUT_DIR

from engine.checkpoint import Checkpoint
from engine.expression import parse
from engine.fsa import skeleton
from engine import failed_patterns as fplib


def main() -> None:
    rejects_path = OUTPUT_DIR / "rejects.jsonl"
    if not rejects_path.exists():
        print("无 rejects.jsonl,退出。")
        sys.exit(1)
    fplib._reset(lib={})                       # 全量重建,不吃进程内缓存
    max_iter = 0
    n_rows = n_uniq = n_parse_err = 0
    seen: set[str] = set()
    with open(rejects_path, encoding="utf-8") as f:
        for line in f:
            n_rows += 1
            r = json.loads(line)
            h = r.get("hash")
            if h in seen:
                continue
            seen.add(h)
            n_uniq += 1
            it = r.get("iter", 0)
            max_iter = max(max_iter, it)
            disp = r.get("disp")
            try:
                skel = skeleton(parse(r["expr"]))
            except Exception:                  # noqa: BLE001  历史表达式解析失败 → 跳过
                n_parse_err += 1
                continue
            if disp in ("stored", "replaced"):
                fplib.record_stored(skel, it)
            else:
                fplib.record_reject(skel, disp, r.get("reasons") or [], it)
    # checkpoint 当前库存也计成功史(含无 rejects 记录的老因子)
    cp = Checkpoint.load(str(OUTPUT_DIR / "checkpoint.json"))
    for fac in cp.stored_factors:
        try:
            fplib.record_stored(skeleton(parse(fac["expr"])), max_iter)
        except Exception:                      # noqa: BLE001
            continue
    fplib.save(max_iter)
    print(f"回填完成: rejects.jsonl {n_rows} 行 / 唯一hash {n_uniq} / 解析失败 {n_parse_err}")
    print(fplib.summary_line())


if __name__ == "__main__":
    main()

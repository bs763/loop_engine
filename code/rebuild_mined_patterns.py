# -*- coding: utf-8 -*-
"""一次性回填累计退休账本(用户 2026-08-27):从 rejects.jsonl 全部历史入库事件
(含被保优淘劣替换出库的,disp=stored/replaced)按 (骨架, 代际) 记账。

跑法:  uv run python code/rebuild_mined_patterns.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "code")

from engine import mined_patterns as mplib
from engine.expression import parse

def main() -> None:
    mplib._reset(lib={})                          # 全量重建
    n = n_err = 0
    with open("output/rejects.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["disp"] not in ("stored", "replaced"):
                continue
            try:
                node = parse(r["expr"])
            except Exception:                     # noqa: BLE001
                n_err += 1
                continue
            mplib.record(node, r.get("iter", 0))
            n += 1
    mplib.save()
    print(f"回填完成: {n} 次入库事件(解析失败 {n_err})")
    print(mplib.summary_line())


if __name__ == "__main__":
    main()

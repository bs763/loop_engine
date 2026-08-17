# -*- coding: utf-8 -*-
"""打印因子库状态摘要(读 output/checkpoint.json)。供人查看 / hook 调用。

跑法:  uv run --directory factor_loop_engine code/lib_status.py
"""
from __future__ import annotations

import json

import numpy as np

from paths import OUTPUT_DIR

CKPT = OUTPUT_DIR / "checkpoint.json"


def _corr_report(factors: list) -> None:
    """两两 IC 相关体检(2026-08-17 加,#9 是入库门槛而非持有门槛,入库后相关会漂移——
    0007 与第一名 0.78 即被坏数据掩盖的存量违规)。≥0.7 打印明细供处置。"""
    sf = [f for f in factors if f.get("ic_series")]
    n = len(sf)
    if n < 2:
        return
    s = [np.asarray(f["ic_series"], dtype=float) for f in sf]
    hi = []
    for i in range(n):
        for j in range(i + 1, n):
            m = min(len(s[i]), len(s[j]))
            c = float(np.corrcoef(s[i][-m:], s[j][-m:])[0, 1])
            if abs(c) >= 0.7:
                hi.append((abs(c), c, sf[i]["expr"], sf[j]["expr"]))
    print(f"相关性体检: 两两 {n * (n - 1) // 2} 对,≥0.7 共 {len(hi)} 对")
    for _a, c, e1, e2 in sorted(hi, reverse=True):
        print(f"  {c:+.3f}  [{e1[:40]}] × [{e2[:40]}]")


def main() -> None:
    if not CKPT.exists():
        print("因子库为空(无 checkpoint.json)。")
        return
    data = json.loads(CKPT.read_text(encoding="utf-8"))
    iteration = data.get("iteration", 0)
    tested = len(data.get("tested_hashes", []))
    factors = data.get("stored_factors", [])
    print(f"迭代={iteration}  已测={tested}  入库={len(factors)}")
    if factors:
        # 按 IC 均值降序,展示 Top 5
        ranked = sorted(factors, key=lambda f: f.get("metrics", {}).get("ic_mean", 0),
                        reverse=True)[:5]
        print("Top 5(按 |IC|):")
        for f in ranked:
            m = f.get("metrics", {})
            print(f"  IC={m.get('ic_mean', 0):.4f} 多空年化={m.get('ls_annual', 0):.2%} "
                  f"夏普={m.get('ls_sharpe', 0):.2f} Calmar={m.get('calmar', 0):.2f} | {f.get('expr')}")
        _corr_report(factors)


if __name__ == "__main__":
    main()

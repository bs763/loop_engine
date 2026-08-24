# -*- coding: utf-8 -*-
"""打印因子库状态摘要(读 output/checkpoint.json)。供人查看 / hook 调用。

跑法:  uv run --directory factor_loop_engine code/lib_status.py
"""
from __future__ import annotations

import json

import numpy as np

from paths import OUTPUT_DIR

CKPT = OUTPUT_DIR / "checkpoint.json"


def oos_health(factors: list) -> str:
    """OOS 系统性崩塌体检(用户 2026-08-18 拍板:崩塌 → 立即停 loop 向用户汇报)。

    崩塌定义(带 OOS 存档的因子 ≥5 个时评估):
      中位 OOS IC ≤ 0,或 OOS IC 为负的因子占比 ≥ 50%。
    正常时返回体检行;崩塌时返回 OOS ALERT(编排器见 ALERT 必须停止续链)。
    """
    ics = [f["oos_metrics"].get("ic_mean") for f in factors if f.get("oos_metrics")]
    ics = [float(x) for x in ics if x is not None]
    if len(ics) < 5:
        return f"OOS体检: 样本不足({len(ics)}/5)"
    ics_sorted = sorted(ics)
    med = ics_sorted[len(ics) // 2] if len(ics) % 2 else (ics_sorted[len(ics) // 2 - 1] + ics_sorted[len(ics) // 2]) / 2
    neg = sum(1 for x in ics if x < 0) / len(ics)
    if med <= 0 or neg >= 0.5:
        return (f"OOS ALERT: 系统性崩塌(中位OOS IC={med:+.4f}, 负占比={neg:.0%})"
                f"→ 立即停止 loop 并向用户汇报!")
    return f"OOS体检: n={len(ics)} 中位OOS IC={med:+.4f} 负占比={neg:.0%}(正常)"


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
        # IS vs OOS 对比(样本外只报告,不参与筛选;老因子无 oos_metrics 则跳过)
        oos_rows = [(f, f["oos_metrics"]) for f in factors if f.get("oos_metrics")]
        if oos_rows:
            print(f"IS→OOS 衰减(样本外,OOS_END~):")
            for f, om in oos_rows:
                im = f.get("metrics", {})
                print(f"  IC {im.get('ic_mean', 0):+.3f}→{om.get('ic_mean', float('nan')):+.3f}  "
                      f"夏普 {im.get('ls_sharpe', 0):.2f}→{om.get('ls_sharpe', float('nan')):.2f}  "
                      f"单调 {im.get('monotonicity', 0):.2f}→{om.get('monotonicity', float('nan')):.2f}"
                      f" | {f.get('expr', '')[:46]}")
        _corr_report(factors)
        print(oos_health(factors))
    # 失败模式库体检(用户 2026-08-24:全灭/占位骨架计数,无库文件 → 提示回填)
    from engine import failed_patterns as fplib
    if (OUTPUT_DIR / "failed_patterns.json").exists():
        print(fplib.summary_line())
    else:
        print("失败模式库: 未建(可跑 code/rebuild_failed_patterns.py 回填历史)")


if __name__ == "__main__":
    main()

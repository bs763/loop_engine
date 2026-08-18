# -*- coding: utf-8 -*-
"""全库体检:用完整检测机制复查库存因子(2026-08-17)。

检查项(全部确定性 + 可选 LLM):
  1. 结构:审查五道(截面折叠/同质退化/跨量纲/最小深度/过度平滑/极值嵌套)
  2. 覆盖率防线:任一月覆盖 < 前后各 12 个月中位 ×60%(需字段面板)
  3. 同构家族:任一 ≥4 节点子树骨架出现在 ≥2 个其它库存因子
  4. 相关性:两两 IC 相关 ≥0.7(违规)/ ≥0.65(贴线观察)
  5. 指标阈值:库内 7 项记录值复核(IC/ICIR/年化/夏普/Calmar/多头超额/单调性;
     末年夏普与滚动 9/12 月需 NAV 重算,不在静态体检范围)
  6. LLM 终审:glm-5.3 经济学把关(可 --no-llm 跳过)

跑法: uv run code/health_check.py [--no-llm]
处置(移除/更新)由用户拍板;本工具只报告。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from engine import review
from engine.expression import parse
from filters import _subtree_skeletons
from loop_orchestrate import _coverage_reason
from paths import OUTPUT_DIR

_THRESHOLDS = [
    ("IC>0.03", lambda m: abs(m.get("ic_mean", 0)) > 0.03),
    ("ICIR>0.3", lambda m: m.get("icir", 0) > 0.30),
    ("年化>0", lambda m: m.get("ls_annual", 0) > 0),
    ("夏普>0.5", lambda m: m.get("ls_sharpe", 0) > 0.5),
    ("Calmar>1", lambda m: m.get("calmar", 0) > 1.0),
    ("多头超额>0", lambda m: m.get("long_excess_annual", 0) > 0),
    ("单调>0.85", lambda m: m.get("monotonicity", 0) > 0.85),
]


def _pairwise_hi(factors: list, lo: float) -> list:
    sf = [f for f in factors if f.get("ic_series")]
    out = []
    for i in range(len(sf)):
        for j in range(i + 1, len(sf)):
            a, b = np.asarray(sf[i]["ic_series"], float), np.asarray(sf[j]["ic_series"], float)
            m = min(len(a), len(b))
            c = float(np.corrcoef(a[-m:], b[-m:])[0, 1])
            if abs(c) >= lo:
                out.append((round(abs(c), 3), sf[i]["hash"][:8], sf[j]["hash"][:8]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="全库体检")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 终审(只跑确定性检查)")
    args = ap.parse_args()

    cp = json.loads((OUTPUT_DIR / "checkpoint.json").read_text(encoding="utf-8"))
    factors = cp["stored_factors"]
    print(f"=== 全库体检:{len(factors)} 个因子,检测项:结构/覆盖/家族/相关/阈值"
          f"{'/LLM终审' if not args.no_llm else '(LLM 跳过)'} ===")

    panels = None
    if not args.no_llm:
        from llm.mechanisms import review_expression
        from llm.settings import review_provider
        llm = review_provider()
    from run_round_cli import _real_panels
    panels = _real_panels()

    subs_all = []
    for f in factors:
        try:
            subs_all.append(_subtree_skeletons(parse(f["expr"])))
        except Exception:  # noqa: BLE001
            subs_all.append(set())
    hi_pairs = _pairwise_hi(factors, 0.70)
    watch_pairs = [p for p in _pairwise_hi(factors, 0.65) if p not in hi_pairs]
    corr70 = set()
    for _c, h1, h2 in hi_pairs:
        corr70.update((h1, h2))

    n_flag = 0
    for idx, f in enumerate(factors):
        flags = []
        # 1) 结构
        try:
            t, reason = review.apply(parse(f["expr"]))
            if t is None:
                flags.append(f"结构:{reason.split('(')[0]}")
        except Exception as e:  # noqa: BLE001
            flags.append(f"结构:解析失败({e})")
        # 2) 覆盖率
        try:
            panel = None
            from engine.expression import evaluate
            panel = evaluate(parse(f["expr"]), panels)
            cov = _coverage_reason(panel)
            if cov:
                flags.append(cov.replace("ValueError: ", "覆盖:"))
        except Exception as e:  # noqa: BLE001
            flags.append(f"覆盖:求值失败({type(e).__name__})")
        # 3) 同构家族(与其它因子的共享子树)
        fam = 0
        for j, other in enumerate(factors):
            if j == idx:
                continue
            if subs_all[idx] & subs_all[j]:
                fam += 1
        # 家族计的是「与多少个其它因子共享任一≥4节点子树」,准入口径按骨架计数,此处报观察值
        # 4) 相关
        if f["hash"][:8] in corr70:
            flags.append("相关:≥0.70 违规对")
        # 5) 指标阈值
        m = f.get("metrics") or {}
        fails = [name for name, ok in _THRESHOLDS if not ok(m)]
        if fails:
            flags.append("阈值:" + ",".join(fails))
        # 6) LLM 终审
        verdict = ""
        if not args.no_llm:
            accept, why = review_expression(llm, parse(f["expr"]), metrics=f.get("metrics"))
            verdict = "终审:ACCEPT" if accept else "终审:REJECT"
            if not accept:
                flags.append(f"终审拒:{why[:60]}")
        status = "PASS " if not flags else "FLAG "
        if flags:
            n_flag += 1
        print(f"[{idx + 1:2d}] {status}{verdict:>14s} 家族共享{fam:2d} | {f['expr'][:56]}")
        for fl in flags:
            print(f"      └ {fl}")

    print(f"\n汇总:{len(factors)} 检完,FLAG {n_flag} 个;"
          f"相关违规对 {len(hi_pairs)}(≥0.70)/ {len(watch_pairs)}(0.65~0.70 观察)")


if __name__ == "__main__":
    main()

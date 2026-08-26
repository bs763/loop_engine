# -*- coding: utf-8 -*-
"""重准入脚本(用户 2026-08-25):对曾被终审误拒的候选重走标准管线。

严格按 run_round 的入库路径:回测 → #1-15 机器过滤(含保优淘劣替换)→ LLM 终审 → OOS 存档 → 入库。
终审拒绝的默认不入库;--force 时用户豁免 #16(终审照跑、留审计记录,但不拦截)。

跑法:  uv run python code/readmit.py [--force] "表达式1" "表达式2" ...
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "code")

import numpy as np

from backtest.alphalab_adapter import AlphalabEvaluator
from engine.checkpoint import Checkpoint
from engine.config import OOS_END, OOS_START
from engine.expression import evaluate, parse
from engine.fsa import skeleton
from engine import failed_patterns as fplib
from filters import apply_filters
from llm.mechanisms import family_of, review_expression
from llm.settings import generation_provider  # noqa: F401  (加载 .env)
from llm.settings import review_provider
from loop_orchestrate import _metrics_summary, restore_fsa
from paths import OUTPUT_DIR, PROJECT_ROOT
from run_round_cli import _real_panels

CKPT = OUTPUT_DIR / "checkpoint.json"
ALPHACFG = str(PROJECT_ROOT / "config" / "alphalab.yaml")


def main() -> None:
    force = "--force" in sys.argv
    exprs = [a for a in sys.argv[1:] if a != "--force"]
    if not exprs:
        print("用法: uv run python code/readmit.py [--force] \"表达式\" ...")
        sys.exit(1)
    panels = _real_panels()
    evaluator = AlphalabEvaluator(horizon=5, config_yaml=ALPHACFG)
    oos_evaluator = AlphalabEvaluator(horizon=5, config_yaml=ALPHACFG,
                                      window=(OOS_START, OOS_END))
    rev = review_provider()
    cp = Checkpoint.load(str(CKPT))
    fsa = restore_fsa(cp)
    it = cp.iteration

    for expr in exprs:
        node = parse(expr)
        node.validate()
        h = node.expr_hash()
        print(f"\n=== 重审: {expr[:70]}… (hash={h[:8]})" if len(expr) > 70
              else f"\n=== 重审: {expr} (hash={h[:8]})")
        existing = [f for f in cp.stored_factors if f["hash"] == h]
        if existing:
            print("  已在库,跳过。")
            continue
        panel = evaluate(node, panels)
        m = evaluator.evaluate(panel, name="ra_" + h[:10])
        fr = apply_filters(m, fsa=fsa, node=node,
                           stored_factors=cp.stored_factors, expr_hash=h)
        if not fr.passed:
            print(f"  [NG] #1-15 机器关未过: {fr.reasons[:3]}")
            continue
        print(f"  [OK] #1-15 机器关全过(IS: IC={m.ic_mean:+.4f} 夏普={m.ls_sharpe:.2f} "
              f"Calmar={m.calmar:.2f} 单调={m.monotonicity:.2f})")
        accept, why = review_expression(rev, node, metrics=m)
        with open(OUTPUT_DIR / "final_review_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"iter": it, "expr": expr, "accept": accept,
                                "raw": why[:300], "readmit": True},
                               ensure_ascii=False) + "\n")
        if not accept and not force:
            print(f"  [NG] #16 终审拒: {why[:120]}")
            continue
        if not accept:
            print(f"  [OK] #16 用户豁免(终审拒因已留档): {why[:100]}")
        else:
            print(f"  [OK] #16 终审通过: {why[:80]}")
        # 保优淘劣替换(用户 2026-08-26:相关超线但全面更优 → 移除在位者)
        if fr.replace_hashes:
            keep = []
            for f in cp.stored_factors:
                if f.get("hash") in fr.replace_hashes:
                    old_skel = f.get("skeleton")
                    if old_skel:
                        fsa.counts[old_skel] = fsa.counts.get(old_skel, 1) - 1
                        if fsa.counts[old_skel] <= 0:
                            del fsa.counts[old_skel]
                else:
                    keep.append(f)
            replaced_n = len(cp.stored_factors) - len(keep)
            cp.stored_factors = keep
            print(f"  [REPLACE] 保优淘劣: 移除 {replaced_n} 个在位者 {fr.replace_hashes}")
        oos = None
        try:
            mo = oos_evaluator.evaluate(evaluate(node, panels), name="roos_" + h[:8])
            oos = _metrics_summary(mo)
        except Exception as e:  # noqa: BLE001
            print(f"  (OOS 评测失败不阻塞: {e})")
        cp.add_factor({
            "expr": expr, "hash": h, "skeleton": skeleton(node),
            "ic_series": m.ic_series, "ls_ret":
                np.diff(np.asarray(m.ls_nav, dtype=float)).tolist(),
            "metrics": _metrics_summary(m), "oos_metrics": oos,
            "family": family_of(h),
        })
        fsa.observe_tree(node)
        fplib.record_stored(node, it)
        with open(OUTPUT_DIR / "rejects.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"iter": it, "hash": h, "expr": expr,
                                "disp": "stored", "reasons": [
                                    f"readmit{'-force(用户豁免#16)' if force else ''}: "
                                    f"替换{len(fr.replace_hashes)}" if fr.replace_hashes else
                                    "readmit: 终审误拒修正后重准入"]},
                               ensure_ascii=False) + "\n")
        print(f"  [STORED] 入库(OOS: IC={oos['ic_mean']:+.4f} 夏普={oos['ls_sharpe']:.2f})"
              if oos else "  ✅ 入库(OOS 无存档)")
    cp.capture(fsa=fsa)
    cp.save()
    fplib.save(it)
    print(f"\n库存: {len(cp.stored_factors)} 个(checkpoint iter={it})")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""按 2026-08-17 结构先验清洗库存因子(审查过滤5 + 过滤15 的存量治理)。

两步:
  1) 结构违规绝对移除(审查过滤5 口径):平滑嵌平滑 / 极值嵌极值;
  2) 同构子树家族贪心裁剪(过滤15 口径):任一 ≥MIN_NODES 节点子树骨架在库中出现 >CAP 个 →
     该族按综合分(filters._quality:IC/ICIR/单调性/多头超额)保前 CAP 个,移除其余;迭代至无超限。

跑法:
  uv run code/clean_library.py            # dry-run(只打印方案,不动库)
  uv run code/clean_library.py --apply    # 真清洗:备份 → 改 checkpoint → 同步 FSA 计数
清洗后建议跑 uv run code/export_factors.py 同步导出 parquet(自动清残留)。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter

from engine import review
from engine.checkpoint import Checkpoint
from engine.config import FAMILY_SUBTREE_CAP, FAMILY_SUBTREE_MIN_NODES
from engine.expression import parse
from filters import _quality, _subtree_skeletons
from paths import OUTPUT_DIR


def _structural_reason(f: dict) -> str | None:
    """审查过滤5 违规(过度平滑/极值嵌套)→ 返回原因;其余(None=通过/其它拒绝类)不算。"""
    try:
        node = parse(f["expr"])
    except Exception as e:  # noqa: BLE001
        return f"parse_error:{e}"
    _t, reason = review.apply(node)
    if reason and ("oversmoothed" in reason or "extreme_nesting" in reason):
        return reason
    return None


def _fam_subs(f: dict) -> set[str]:
    try:
        return _subtree_skeletons(parse(f["expr"]))
    except Exception:  # noqa: BLE001
        return set()


def _quality_of(f: dict) -> float:
    m = f.get("metrics") or {}
    return _quality(m.get("ic_mean", 0), m.get("icir", 0),
                    m.get("monotonicity", 0), m.get("long_excess_annual", 0))


def clean(factors: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """返回 (保留列表, 移除台账 [(因子, 原因)])。"""
    kept, removed = [], []
    # 1) 结构违规(绝对移除)
    for f in factors:
        r = _structural_reason(f)
        if r:
            removed.append((f, f"结构违规:{r}"))
        else:
            kept.append(f)
    # 2) 同构子树家族贪心裁剪(每族保综合分前 CAP 个)
    while True:
        subs = [(f, _fam_subs(f)) for f in kept]
        counts: Counter = Counter()
        for _f, sks in subs:
            for sk in sks:
                counts[sk] += 1
        over = {sk: c for sk, c in counts.items() if c > FAMILY_SUBTREE_CAP}
        if not over:
            break
        sk = max(over, key=lambda s: over[s])            # 最超限的骨架先裁
        members = [f for f, sks in subs if sk in sks]
        members.sort(key=lambda f: (_quality_of(f), f["expr"]), reverse=True)
        for f in members[FAMILY_SUBTREE_CAP:]:
            kept.remove(f)
            removed.append((f, f"同构家族裁剪(共{over[sk]}个保{FAMILY_SUBTREE_CAP}):{sk[:44]}"))
    return kept, removed


def main() -> None:
    ap = argparse.ArgumentParser(description="按结构先验清洗库存因子")
    ap.add_argument("--checkpoint", default=str(OUTPUT_DIR / "checkpoint.json"))
    ap.add_argument("--apply", action="store_true", help="真清洗(默认 dry-run)")
    args = ap.parse_args()

    cp = Checkpoint.load(args.checkpoint)
    n0 = len(cp.stored_factors)
    kept, removed = clean(cp.stored_factors)
    print(f"库存 {n0} → 保留 {len(kept)},移除 {len(removed)}")
    for f, r in removed:
        m = f.get("metrics") or {}
        print(f"  移除 [{_quality_of(f):.2f}分 ic={m.get('ic_mean', 0):.3f}] {f['expr'][:66]}")
        print(f"        └ {r}")
    # 终检:保留集应同时满足过滤5与家族上限
    bad = [f["expr"][:50] for f in kept if _structural_reason(f)]
    print(f"\n终检:保留集结构违规 {len(bad)} 个{' ✗' if bad else ' ✓'}")

    if not args.apply:
        print("\n(dry-run,未落盘;加 --apply 执行)")
        return

    backup = cp.path.with_name(cp.path.name + ".bak-20260817-preclean")
    shutil.copy2(cp.path, backup)
    counts = dict(cp.fsa_state.get("counts", {}))
    for f, _r in removed:                      # FSA 整树骨架计数同步扣减
        sk = f.get("skeleton")
        if sk and sk in counts:
            counts[sk] -= 1
            if counts[sk] <= 0:
                del counts[sk]
    cp.fsa_state = {"counts": counts}
    cp.stored_factors = kept
    cp.save()
    ledger = OUTPUT_DIR / "library_clean_20260817.json"
    with open(ledger, "w", encoding="utf-8") as fh:
        json.dump({"kept": len(kept),
                   "removed": [{"expr": f["expr"], "hash": f.get("hash"), "reason": r,
                                "metrics": f.get("metrics")} for f, r in removed]},
                  fh, ensure_ascii=False, indent=1)
    print(f"\n已落盘:{cp.path}(备份 → {backup.name});移除台账 → {ledger.name}")


if __name__ == "__main__":
    main()

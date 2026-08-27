# -*- coding: utf-8 -*-
"""累计退休制(用户 2026-08-27)—— 骨架开采额度按字段代际分账。

与失败模式库(failed_patterns,失败端)对偶:本库记【成功端】的历史开采量。
背景:保优淘劣使 FSA#10 并发口径永不触发(替换出库→计数减,全历史 0 次);
热门子树插件被反复开采无记忆(zscore(std(rank_cs(FLD),N)) 历史累计 60 次)。

口径:
  - 计数单位 = 入库事件(含被替换出库的,disp=stored/replaced),每次 +1,替换不返还;
  - 代际分账(用户要求新字段有包容度):骨架计数键 = (骨架, 代际)。
    价量账(pv)继承全部历史欠账;基本面账(fund)从实际小计数起步,有自己的额度;
    混血子树(任一叶子为基本面字段)记【fund 新额度】,不消耗 pv 欠账(用户 2026-08-27);
  - 退休判定:候选的整树骨架或任一 ≥4 节点子树骨架,在其叶子所属代际的账上
    累计 ≥ 上限(MINED_TREE_CAP / MINED_SUBTREE_CAP)→ 该结构对该代际退休,
    审查端确定性拒(review:mined_out),新候选永久去重。

存储 output/mined_patterns.json:{"骨架|代际": 次数}(原子写)。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from paths import OUTPUT_DIR

from engine.config import (FUND_FIELDS, MINED_SUBTREE_CAP, MINED_TREE_CAP,
                           FAMILY_SUBTREE_MIN_NODES)

_PATH: Path = OUTPUT_DIR / "mined_patterns.json"
_lib: dict | None = None


def _load() -> dict:
    global _lib
    if _lib is None:
        try:
            _lib = json.loads(_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  首次/损坏 → 空
            _lib = {}
    return _lib


def _reset(path: Path | None = None, lib: dict | None = None) -> None:
    """测试用:换路径/清缓存。"""
    global _PATH, _lib
    if path is not None:
        _PATH = path
    _lib = lib if lib is not None else {}


# ---- 结构枚举 ----

def _size(n) -> int:
    return 1 + sum(_size(c) for c in n.children)


def _cohort(node) -> str:
    """子树的代际(单值):任一叶子是基本面字段 → fund(混血记新额度,用户 2026-08-27:
    "有一个是新的有一个是老的,这个子树应该算做新的额度"——混血是基本面探索的已验证
    形态,不应消耗 pv 的历史欠账);全价量 → pv。"""
    for f in node.fields():
        if f in FUND_FIELDS:
            return "fund"
    return "pv"


def _subtrees(node, min_nodes: int = FAMILY_SUBTREE_MIN_NODES):
    """枚举 >=min_nodes 节点的子树(不含整树;整树单独由 record/is_mined_out 处理)。"""
    out = []
    def rec(n):
        for c in n.children:
            if _size(c) >= min_nodes:
                out.append(c)
            rec(c)
    rec(node)
    return out


# ---- 记录 / 判定 ----

def record(node, iteration: int) -> None:
    """入库事件记账(整树 T| 前缀,子树 S| 前缀,代际分账)。

    整树自身 >=4 节点时同时记一笔 S|(它可能作为子树出现在未来的更大组合里)。"""
    from engine.fsa import skeleton
    lib = _load()
    t_skel = skeleton(node)
    tc = _cohort(node)
    lib[f"T|{t_skel}|{tc}"] = lib.get(f"T|{t_skel}|{tc}", 0) + 1
    if _size(node) >= FAMILY_SUBTREE_MIN_NODES:
        lib[f"S|{t_skel}|{tc}"] = lib.get(f"S|{t_skel}|{tc}", 0) + 1
    for sub in _subtrees(node):
        s_skel = skeleton(sub)
        sc = _cohort(sub)
        lib[f"S|{s_skel}|{sc}"] = lib.get(f"S|{s_skel}|{sc}", 0) + 1


def is_mined_out(node) -> str | None:
    """退休判定:候选整树或任一 >=4 节点子树,在其代际账上累计达上限 → 返回拒因。"""
    from engine.fsa import skeleton
    lib = _load()
    if not lib:
        return None
    t_skel = skeleton(node)
    tc = _cohort(node)
    if lib.get(f"T|{t_skel}|{tc}", 0) >= MINED_TREE_CAP[tc]:
        return f"mined_out_tree({t_skel[:50]}…代际{tc})"
    for sub in _subtrees(node):
        s_skel = skeleton(sub)
        sc = _cohort(sub)
        if lib.get(f"S|{s_skel}|{sc}", 0) >= MINED_SUBTREE_CAP[sc]:
            return f"mined_out_subtree({s_skel[:50]}…代际{sc})"
    return None


def save() -> None:
    """原子落盘。"""
    lib = _load()
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False)
    os.replace(tmp, _PATH)


def summary_line() -> str:
    """体检行:已退休的结构数(按代际)。"""
    lib = _load()
    ret = {"pv": 0, "fund": 0}
    for k, v in lib.items():
        kind, cohort = k[0], k.rsplit("|", 1)[1]     # 键格式 T|骨架|代际 / S|骨架|代际
        cap = MINED_TREE_CAP if kind == "T" else MINED_SUBTREE_CAP
        if v >= cap[cohort]:
            ret[cohort] += 1
    return f"累计退休制: 账目{len(lib)}条 已退休 价量{ret['pv']} 基本面{ret['fund']}"

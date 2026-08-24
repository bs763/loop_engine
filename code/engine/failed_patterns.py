# -*- coding: utf-8 -*-
"""失败模式库(M9,用户 2026-08-24)—— 结构级失败经验 → 生成端回流。

与 FSA(#10 用库存骨架防结构拥挤)对偶:本库用失败端防结构浪费。
实测样本:18777 个唯一被拒 hash 中,123 个骨架 ≥10 连败;其中 72 个的拒因**仅有**
#9 IC相关 / #15 同构家族 —— 那是库内在位强因子占位(保优淘劣挑战者路径),**不算死证据**;
真正全灭的 45 个(裸浅树、多字段组合连败)才回流生成端规避。

内因口径(用户 2026-08-24 拍板):
  - 占位灭:filter_reject 的 reasons 全部 ∈ {#9, #15} → occupied+1,不回流;
  - 内因灭:review 结构拒 / 回测异常 / 过滤规则 1-8、12-14、16 → fails+1,可回流;
  - 判全灭:fails ≥ DEAD_MIN_FAILS 且 stored == 0(同骨架曾有因子入库 → 该结构可行,不判死;
    被替换出库也不扣减——成功史是永久的)。

防 Goodhart 边界:回流的是骨架 → 连败计数 + 头号内因**类别**,不含任何指标数值
(与 family_notes 同一原则,loop_orchestrate「指标不回流生成端」)。

存储 output/failed_patterns.json(原子写):
  {"version": 1, "updated_iter": N,
   "patterns": {骨架: {"fails", "occupied", "stored", "last_iter", "top_rules": {规则号: n}}}}
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from paths import OUTPUT_DIR

from engine.config import DEAD_MIN_FAILS, DEAD_PROMPT_TOP_K
from engine.fsa import skeleton as _skeleton
from engine.expression import Node

# 占位类规则号:拒因全是这些 → 在位因子挡路,不算死证据(#9 IC相关 / #15 同构家族)
_OCCUPIED_RULES = {"9", "15"}

# 展示用规则标签(体检行 / prompt 注入)
RULE_LABELS = {
    "1": "IC弱", "2": "某年负", "3": "年化负", "4": "夏普低", "5": "末年夏普低",
    "6": "Calmar低", "7": "近9月负", "8": "近12月负", "9": "相关性高", "10": "FSA结构",
    "11": "失败模式库", "12": "多头超额负", "13": "单调性低", "14": "ICIR低",
    "15": "同构家族", "16": "LLM终审", "review": "审查结构拒", "bt_err": "回测异常",
}

_PATH: Path = OUTPUT_DIR / "failed_patterns.json"
_lib: dict | None = None          # 惰性单例(进程内)


def failed_patterns(path: Path | None = None) -> dict:
    """惰性加载 {骨架: {fails, occupied, stored, last_iter, top_rules}},损坏/缺失 → 空。"""
    global _lib
    if path is not None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    if _lib is None:
        try:
            data = json.loads(_PATH.read_text(encoding="utf-8"))
            _lib = data.get("patterns", {})
        except Exception:  # noqa: BLE001  首次/损坏 → 空
            _lib = {}
    return _lib


def _reset(path: Path | None = None, lib: dict | None = None) -> None:
    """测试用:换路径/清缓存。lib={} 显式清空;lib=None → 置回惰性(下次从磁盘重载)。"""
    global _PATH, _lib
    if path is not None:
        _PATH = path
    _lib = lib   # None → 惰性重载;{} → 空库


def _entry(skel: str, iteration: int) -> dict:
    lib = failed_patterns()
    e = lib.setdefault(skel, {"fails": 0, "occupied": 0, "stored": 0,
                              "last_iter": iteration, "top_rules": {}})
    e["last_iter"] = max(e.get("last_iter", 0), iteration)
    return e


def _rule_numbers(reasons: list[str]) -> set[str]:
    """从 reason 字符串提取规则号(如 "9.IC相关性=…" → "9");无编号 → 空。"""
    out = set()
    for r in reasons or []:
        m = re.match(r"(\d+)\.", str(r))
        if m:
            out.add(m.group(1))
    return out


def record_reject(node_or_skel: "Node | str", disp: str, reasons: list[str],
                  iteration: int) -> None:
    """记录一次拒绝(同 hash 终身只拒一次,由 tested_hashes 保证,无需在此去重)。

    内因判定:filter_reject 且拒因全为占位规则(#9/#15)→ occupied;其余(含 review/
    backtest_error)→ fails,非占位规则号进 top_rules。
    """
    skel = node_or_skel if isinstance(node_or_skel, str) else _skeleton(node_or_skel)
    if disp == "filter_reject":
        rules = _rule_numbers(reasons)
        if rules and rules <= _OCCUPIED_RULES:
            _entry(skel, iteration)["occupied"] += 1
            return
        e = _entry(skel, iteration)
        e["fails"] += 1
        for k in sorted(rules - _OCCUPIED_RULES):
            e["top_rules"][k] = e["top_rules"].get(k, 0) + 1
        return
    # review_reject(结构拒)/ backtest_error(确定性结构缺陷)→ 直接死证据
    e = _entry(skel, iteration)
    e["fails"] += 1
    key = "review" if disp == "review_reject" else "bt_err"
    e["top_rules"][key] = e["top_rules"].get(key, 0) + 1


def record_stored(node_or_skel: "Node | str", iteration: int) -> None:
    """记录一次入库成功(replaced 出库者不扣减——成功史永久,天然不判死)。"""
    skel = node_or_skel if isinstance(node_or_skel, str) else _skeleton(node_or_skel)
    e = _entry(skel, iteration)
    e["stored"] += 1


def dead_patterns(min_fails: int = DEAD_MIN_FAILS) -> list[tuple[str, dict]]:
    """全灭骨架按内因失败数降序:fails ≥ min_fails 且 0 成功。"""
    lib = failed_patterns()
    dead = [(s, e) for s, e in lib.items()
            if e.get("fails", 0) >= min_fails and e.get("stored", 0) == 0]
    return sorted(dead, key=lambda kv: -kv[1]["fails"])


def dead_skeletons(min_fails: int = DEAD_MIN_FAILS) -> set[str]:
    return {s for s, _ in dead_patterns(min_fails)}


def top_rule_label(info: dict) -> str:
    """头号内因类别(展示用):规则号 → 中文标签。"""
    tr = info.get("top_rules") or {}
    if not tr:
        return "未知"
    k = max(tr, key=tr.get)
    return RULE_LABELS.get(k, f"规则{k}")


def prompt_block(top_k: int = DEAD_PROMPT_TOP_K) -> str:
    """生成 prompt 注入段(空库 → 空串)。只含骨架 + 连败数 + 内因类别,无指标数值。"""
    dead = dead_patterns()
    if not dead:
        return ""
    lines = [f"- {s} ×{e['fails']}连败(头号内因:{top_rule_label(e)})"
             for s, e in dead[:top_k]]
    return ("\n【已知全灭结构(以下骨架的任何字段/窗口组合都因内在弱点失败,生成时避开)】\n"
            + "\n".join(lines) + "\n")


def save(iteration: int) -> None:
    """原子落盘(tmp + os.replace)。轮末调用一次;测试经 _reset(path=…) 换路径。"""
    lib = failed_patterns()
    p = _PATH
    data = {"version": 1, "updated_iter": iteration, "patterns": lib}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def summary_line() -> str:
    """体检行:骨架总数 / 全灭数 / 占位骨架数 / Top1。"""
    lib = failed_patterns()
    n_dead = len(dead_patterns())
    n_occ = sum(1 for e in lib.values()
                if e.get("occupied", 0) > 0 and e.get("stored", 0) == 0
                and e.get("fails", 0) < DEAD_MIN_FAILS)
    head = "失败模式库: " + f"骨架{len(lib)} 全灭{n_dead} 占位{n_occ}"
    dead = dead_patterns()
    if dead:
        s, e = dead[0]
        head += f" | Top1: {s}×{e['fails']}({top_rule_label(e)})"
    return head

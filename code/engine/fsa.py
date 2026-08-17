# -*- coding: utf-8 -*-
"""M8 频繁子树规避(Frequent Subtree Avoidance)—— 抑制结构趋同。

两道机制:
  1. **骨架冻结**:把表达式抽象成骨架(字段→FLD、窗口→N,只留算子结构);
     若某骨架在因子库的**支持度(占比)> 15% 且出现 ≥ 2 次** → 冻结,新生成的同骨架候选拒绝。
  2. **同骨架参数变体上限**:同一骨架最多存 5 个(防止同结构换参数刷数量),第 6 个拒绝。

研报实证:450 轮后 overnight 骨架超 15% 被冻结,迫使搜索转向 down_shadow / hl_ratio。
"""
from __future__ import annotations

from collections import Counter

from engine.config import (  # [研报] FSA 阈值
    FSA_MIN_COUNT as MIN_COUNT,
    FSA_PARAM_VARIANT_CAP as PARAM_VARIANT_CAP,
    FSA_SUPPORT_THRESHOLD as SUPPORT_THRESHOLD,
)
from engine.expression import Node


def skeleton(node: Node) -> str:
    """抽象骨架:叶子→FLD,时序窗口→N,保留算子结构。

    例:div(ma(close,20), ma(close,10)) → div(ma(FLD,N), ma(FLD,N))
    """
    if node.is_leaf():
        return "FLD"
    if node.window is not None:  # 时序算子
        return f"{node.op}({skeleton(node.children[0])}, N)"
    return f"{node.op}({', '.join(skeleton(c) for c in node.children)})"


class FSA:
    """频繁子树规避器:维护因子库骨架计数,提供冻结/变体上限检查。"""

    def __init__(self, support_threshold: float = SUPPORT_THRESHOLD,
                 min_count: int = MIN_COUNT, param_variant_cap: int = PARAM_VARIANT_CAP):
        self.counts: Counter[str] = Counter()
        self.support_threshold = support_threshold
        self.min_count = min_count
        self.param_variant_cap = param_variant_cap

    # ---- 维护 ----
    def total(self) -> int:
        return sum(self.counts.values())

    def observe(self, skel: str) -> None:
        """记录一个骨架入库(通常在因子通过筛选入库时调用)。"""
        self.counts[skel] += 1

    def observe_tree(self, node: Node) -> None:
        self.observe(skeleton(node))

    # ---- 查询 ----
    def support(self, skel: str) -> float:
        return self.counts[skel] / self.total() if self.total() else 0.0

    def is_frozen(self, skel: str) -> bool:
        """骨架是否被冻结(支持度>阈值 且 次数≥min_count)。"""
        return self.counts[skel] >= self.min_count and self.support(skel) > self.support_threshold

    def check(self, node: Node) -> tuple[bool, str]:
        """检查候选因子是否被 FSA 拒。返回 (通过, 原因)。"""
        skel = skeleton(node)
        if self.is_frozen(skel):
            return False, f"fsa:frozen_skeleton({skel}, support={self.support(skel):.1%})"
        if self.counts[skel] >= self.param_variant_cap:
            return False, f"fsa:param_variant_cap({skel}, n={self.counts[skel]})"
        return True, ""

    # ---- 检查点持久化 ----
    def state(self) -> dict:
        return {"counts": dict(self.counts)}

    def load_state(self, state: dict) -> None:
        self.counts = Counter(state.get("counts", {}))

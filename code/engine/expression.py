# -*- coding: utf-8 -*-
"""表达式树:叶子=数据字段,内部节点=算子,最大深度 4 层。

表示 s-表达式,如:`zscore(div(ma(close,20), std(sub(high,low),10)))`
  - 叶子:裸字段名 `close`
  - 时序算子:`op(child, n)`  如 `ma(close, 20)`
  - 逐元素算子:`op(left, right)`  如 `sub(high, low)`
  - 截面算子:`op(child)`  如 `zscore(close)`

提供:Node 数据结构、parse/to_str 往返、depth/fields/validate、evaluate(在字段宽表上求值)、
随机生成(冷启动)、哈希(去重)。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from engine.config import MAX_DEPTH, WINDOW_SET  # [研报] 深度;[用户] 窗口离散集
from engine.operators import (
    ALL_OP_NAMES, CS_OP_NAMES, ELEM_OP_NAMES, TS_OP_NAMES, OP_REGISTRY, apply,
)

# 默认随机源(本模块是普通 Python,可用 numpy 随机)
_RNG = np.random.default_rng()


@dataclass
class Node:
    """表达式树节点。

    叶子:op=None, field=<字段名>;内部节点:op=<算子名>, children=[...]。
    时序算子额外带 window=n。
    """
    op: str | None = None
    field: str | None = None
    window: int | None = None
    children: list = dc_field(default_factory=list)

    # ---- 构造便捷方法 ----
    @classmethod
    def leaf(cls, field_name: str) -> "Node":
        return cls(op=None, field=field_name)

    @classmethod
    def ts(cls, op: str, child: "Node", window: int) -> "Node":
        return cls(op=op, window=window, children=[child])

    @classmethod
    def cs(cls, op: str, child: "Node") -> "Node":
        return cls(op=op, children=[child])

    @classmethod
    def elem(cls, op: str, left: "Node", right: "Node") -> "Node":
        return cls(op=op, children=[left, right])

    # ---- 属性 ----
    def is_leaf(self) -> bool:
        return self.op is None

    def depth(self) -> int:
        """算子层数:叶子=0,否则 1+max(子节点深度)。"""
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def fields(self) -> set[str]:
        if self.is_leaf():
            return {self.field}
        out: set[str] = set()
        for c in self.children:
            out |= c.fields()
        return out

    def walk(self):
        """前序遍历所有节点。"""
        yield self
        for c in self.children:
            yield from c.walk()

    # ---- 序列化 ----
    def to_str(self) -> str:
        if self.is_leaf():
            return str(self.field)
        if self.window is not None:  # 时序
            return f"{self.op}({self.children[0].to_str()}, {self.window})"
        return f"{self.op}({', '.join(c.to_str() for c in self.children)})"

    def __repr__(self) -> str:
        return self.to_str()

    def __eq__(self, other) -> bool:
        return isinstance(other, Node) and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())

    # ---- 校验 ----
    def validate(self) -> bool:
        """校验:算子名合法、arity 匹配、时序 window 在合法范围;非法抛 ValueError。"""
        if self.is_leaf():
            if not self.field:
                raise ValueError("叶子缺字段名")
            return True
        if self.op not in OP_REGISTRY:
            raise ValueError(f"未知算子: {self.op}")
        info = OP_REGISTRY[self.op]
        if info["kind"] == "ts":
            if len(self.children) != 1:
                raise ValueError(f"{self.op} 需 1 子节点,得 {len(self.children)}")
            if self.window is None:
                raise ValueError(f"{self.op} 缺 window")
            lo, hi = info["window_range"]
            if not (lo <= self.window <= hi):
                raise ValueError(f"{self.op} window={self.window} 不在 [{lo},{hi}]")
        elif info["kind"] == "elem":
            if len(self.children) != 2:
                raise ValueError(f"{self.op} 需 2 子节点,得 {len(self.children)}")
        else:  # cs
            if len(self.children) != 1:
                raise ValueError(f"{self.op} 需 1 子节点,得 {len(self.children)}")
        for c in self.children:
            c.validate()
        return True

    def expr_hash(self) -> str:
        """表达式哈希(基于 to_str,用于去重)。"""
        return hashlib.sha1(self.to_str().encode()).hexdigest()[:16]


# ============================================================================
# 解析
# ============================================================================

_TOK = re.compile(r"[(),]|[^\s(),]+")


def _tokenize(s: str) -> list[str]:
    return [m.group() for m in _TOK.finditer(s)]


def parse(s: str) -> Node:
    """解析 s-表达式字符串 → Node。"""
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError(f"空表达式: {s!r}")
    node, pos = _parse(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"表达式多余 token: {s!r}")
    return node


def _parse(tokens: list[str], i: int) -> tuple[Node, int]:
    name = tokens[i]
    i += 1
    if i < len(tokens) and tokens[i] == "(":
        i += 1
        args: list = []
        while tokens[i] != ")":
            atom, i = _parse_atom(tokens, i)
            args.append(atom)
            if tokens[i] == ",":
                i += 1
        i += 1  # 消费 ')'
        return _build(name, args), i
    return Node.leaf(name), i


def _parse_atom(tokens: list[str], i: int) -> tuple[object, int]:
    """一个原子:整数字面量 或 嵌套表达式(Node)。"""
    if tokens[i].lstrip("-").isdigit():
        return int(tokens[i]), i + 1
    return _parse(tokens, i)


def _build(name: str, args: list) -> Node:
    if name not in OP_REGISTRY:
        raise ValueError(f"未知算子: {name}")
    info = OP_REGISTRY[name]
    children = [a for a in args if isinstance(a, Node)]
    windows = [a for a in args if isinstance(a, int)]
    if info["kind"] == "ts":
        return Node.ts(name, children[0], windows[0])
    if info["kind"] == "elem":
        return Node.elem(name, children[0], children[1])
    return Node.cs(name, children[0])  # cs


# ============================================================================
# 求值
# ============================================================================

def evaluate(node: Node, field_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """在字段宽表上递归求值,返回宽表面板。叶子消毒:原始数据混入的 ±inf → NaN,
    不让它进任何算子(时序滚动窗口会把 inf 保持整个窗口期,截面算子会被单个 inf 毒化)。"""
    if node.is_leaf():
        if node.field not in field_panels:
            raise KeyError(f"字段面板缺少: {node.field}")
        p = field_panels[node.field]
        return p.replace([np.inf, -np.inf], np.nan)
    info = OP_REGISTRY[node.op]
    child_panels = [evaluate(c, field_panels) for c in node.children]
    if info["kind"] == "ts":
        return apply(node.op, child_panels, window=node.window)
    return apply(node.op, child_panels)


# ============================================================================
# 随机生成(冷启动)
# ============================================================================

# 算子大类权重
_KIND_WEIGHTS = {"ts": 0.5, "elem": 0.3, "cs": 0.2}


def random_tree(fields: list[str], max_depth: int = MAX_DEPTH,
                rng: np.random.Generator | None = None, _depth: int = 0,
                field_weights: np.ndarray | None = None,
                min_depth: int | None = None) -> Node:
    """随机生成表达式树(min_depth ≤ 算子层数 ≤ max_depth)。

    越深越倾向叶子,保证有界;冷启动保证初始广度。
    field_weights 给字段非均匀权重(长度=len(fields),需归一化),用于把生成偏向某些字段(如跳空)。
    min_depth 默认取 config.REVIEW_MIN_DEPTH(用户 2026-08-24:与其让单层树生成后被审查
    确定性拒绝、进失败模式库占位,不如源头不生成——生成端与审查阈值同源,放宽自动跟随)。
    """
    rng = rng if rng is not None else _RNG
    if min_depth is None:
        from engine.config import REVIEW_MIN_DEPTH
        min_depth = REVIEW_MIN_DEPTH
    min_depth = min(min_depth, max_depth)

    def _leaf() -> Node:
        if field_weights is None:
            return Node.leaf(str(rng.choice(fields)))
        return Node.leaf(str(fields[int(rng.choice(len(fields), p=field_weights))]))

    # 越深越倾向取叶子:_depth<min_depth 必取算子(保证至少 min_depth 层);
    # 临近 max_depth 提高叶子概率
    if _depth >= max_depth:
        return _leaf()
    leaf_prob = 0.0 if _depth < min_depth else (0.1 if _depth < max_depth - 1 else 0.5)
    if rng.random() < leaf_prob:
        return _leaf()

    kind = rng.choice(["ts", "elem", "cs"], p=[_KIND_WEIGHTS["ts"], _KIND_WEIGHTS["elem"], _KIND_WEIGHTS["cs"]])
    if kind == "ts":
        opname = str(rng.choice(TS_OP_NAMES))
        lo, hi = OP_REGISTRY[opname]["window_range"]
        grid = [w for w in WINDOW_SET if lo <= w <= hi]
        window = int(rng.choice(grid)) if grid else int(rng.integers(lo, hi + 1))
        return Node.ts(opname, random_tree(fields, max_depth, rng, _depth + 1, field_weights), window)
    if kind == "elem":
        opname = str(rng.choice(ELEM_OP_NAMES))
        return Node.elem(opname,
                         random_tree(fields, max_depth, rng, _depth + 1, field_weights),
                         random_tree(fields, max_depth, rng, _depth + 1, field_weights))
    opname = str(rng.choice(CS_OP_NAMES))
    return Node.cs(opname, random_tree(fields, max_depth, rng, _depth + 1, field_weights))


def expr_hash(node: Node) -> str:
    return node.expr_hash()

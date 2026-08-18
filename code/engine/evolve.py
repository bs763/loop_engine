# -*- coding: utf-8 -*-
"""M3 五维演化引擎:变异 / 交叉 / 参数扰动 / 随机 / LLM。

默认预算(可调):mutate 25% / crossover 25% / perturb 15% / random 15% / llm 20%。
冷启动(parents 为空)→ 全随机,保证初始广度;有父本后切演化模式。
种子选择优先级:从**已入库因子**(过 11 项筛选,质量更高)选父本(由调用方传入 parents)。

所有操作**保结构**:产出必为合法表达式(validate 通过、深度 ≤ max_depth、字段 ⊆ 允许集)。
LLM 分支在阶段 4 接 provider;阶段 3 为 stub(退化为随机)。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from engine.config import EVOLVE_BUDGET, MAX_DEPTH, WINDOW_SET  # [研报] 预算/深度;[用户] 窗口集
from engine.expression import Node, random_tree
from engine.operators import OP_REGISTRY
from engine.perturb import Perturber

# 默认五维预算(M3)——[研报]
DEFAULT_BUDGET = EVOLVE_BUDGET


@dataclass
class EvolveConfig:
    mutate: float = EVOLVE_BUDGET["mutate"]
    crossover: float = EVOLVE_BUDGET["crossover"]
    perturb: float = EVOLVE_BUDGET["perturb"]
    random: float = EVOLVE_BUDGET["random"]
    llm: float = EVOLVE_BUDGET["llm"]
    max_depth: int = MAX_DEPTH

    def __post_init__(self) -> None:
        total = self.mutate + self.crossover + self.perturb + self.random + self.llm
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"五维预算之和必须为 1,当前 {total}")


# ---------------- 树操作工具(不可变) ----------------

def _all_paths(node: Node, prefix: tuple = ()):
    """所有节点路径(根为空 tuple)。"""
    yield prefix
    for i, c in enumerate(node.children):
        yield from _all_paths(c, prefix + (i,))


def _node_at(node: Node, path: tuple) -> Node:
    for i in path:
        node = node.children[i]
    return node


def _with_replaced(node: Node, path: tuple, new: Node) -> Node:
    """返回新树:path 处子树替换为 new。"""
    if not path:
        return new
    i = path[0]
    children = list(node.children)
    children[i] = _with_replaced(children[i], path[1:], new)
    return Node(node.op, node.field, node.window, children)


def _clone(node: Node) -> Node:
    return copy.deepcopy(node)


def _snap_window(n: int, lo: int, hi: int) -> int:
    """把扰动得到的窗口 snap 到 WINDOW_SET ∩ [lo,hi] 中最近的规整值。"""
    cand = [w for w in WINDOW_SET if lo <= w <= hi]
    if not cand:
        return n
    return min(cand, key=lambda w: abs(w - n))


class Evolver:
    """五维演化引擎。"""

    def __init__(self, fields: list[str], config: EvolveConfig | None = None,
                 perturber: Perturber | None = None,
                 rng: np.random.Generator | None = None,
                 llm_provider=None):
        self.fields = list(fields)
        self.cfg = config or EvolveConfig()
        self.perturber = perturber or Perturber()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.llm_provider = llm_provider  # 阶段 4 接入;None → stub
        self._field_usage: dict[str, int] = {}   # 字段→已入库因子中使用次数(引导转向未充分挖掘字段)
        # 字段 boost:把生成额外拉向「隔夜跳空」等未充分挖掘来源(权重乘数,默认跳空优先)
        self._boost_fields: dict[str, float] = {"overnight": 4.0, "intraday": 2.0, "amplitude": 2.0}

    # ---- 父本采样 ----
    def _pick_parent(self, parents: list[Node]) -> Node:
        return parents[int(self.rng.integers(0, len(parents)))]

    def _pick_two(self, parents: list[Node]) -> tuple[Node, Node]:
        if len(parents) >= 2:
            i, j = self.rng.choice(len(parents), size=2, replace=False)
            return parents[int(i)], parents[int(j)]
        p = parents[0]
        return p, p

    def _rand_field(self) -> str:
        # 偏向未充分挖掘字段(1/(1+使用次数)),再乘字段 boost(跳空优先)
        w = self._field_weights()
        return str(self.fields[int(self.rng.choice(len(self.fields), p=w))])

    def _field_weights(self) -> np.ndarray:
        """字段选择权重 = boost / (1 + 使用次数),归一化。"""
        w = np.array([self._boost_fields.get(f, 1.0) / (1 + self._field_usage.get(f, 0))
                      for f in self.fields], dtype=float)
        return w / w.sum()

    def set_field_usage(self, usage: dict[str, int]) -> None:
        """引导字段选择偏向未充分挖掘的字段(usage: 字段 → 已入库因子中使用次数)。"""
        self._field_usage = dict(usage)

    def set_boost_fields(self, boost: dict[str, float]) -> None:
        """设置字段 boost(字段 → 权重乘数),用于把生成额外拉向指定来源(如跳空)。"""
        self._boost_fields = dict(boost)

    # ---- 操作选择 ----
    def _pick_op(self) -> str:
        r = float(self.rng.random())
        cum = 0.0
        for name in ("mutate", "crossover", "perturb", "random", "llm"):
            cum += getattr(self.cfg, name)
            if r < cum:
                return name
        return "llm"

    # ---- 五种操作 ----
    def mutate(self, tree: Node) -> Node:
        """变异:随机选一叶子,或换字段(保骨架换信号源)或长出小子树。"""
        leaf_paths = [p for p in _all_paths(tree) if _node_at(tree, p).is_leaf()]
        path = leaf_paths[int(self.rng.integers(0, len(leaf_paths)))]
        budget = self.cfg.max_depth - len(path)
        if budget <= 0 or float(self.rng.random()) < 0.7:
            new = Node.leaf(self._rand_field())          # 换信号源,骨架不变
        else:
            new = random_tree(self.fields, max_depth=min(2, budget), rng=self.rng,
                              field_weights=self._field_weights())
        return _with_replaced(tree, path, new)

    def crossover(self, a: Node, b: Node) -> Node:
        """交叉:把 b 的一个子树嫁接到 a 的某位置(深度受限)。"""
        a_paths = list(_all_paths(a))
        b_paths = list(_all_paths(b))
        for _ in range(20):
            ap = a_paths[int(self.rng.integers(0, len(a_paths)))]
            bp = b_paths[int(self.rng.integers(0, len(b_paths)))]
            donor = _node_at(b, bp)
            if len(ap) + donor.depth() <= self.cfg.max_depth:
                return _with_replaced(a, ap, _clone(donor))
        # 屡次超深 → 退化为叶子嫁接(保证深度)
        ap = a_paths[int(self.rng.integers(0, len(a_paths)))]
        return _with_replaced(a, ap, Node.leaf(self._rand_field()))

    def perturb_op(self, tree: Node) -> Node:
        """参数扰动:对每个时序算子窗口用 perturber 提议新值(结构不变)。"""
        return self._perturb_rec(tree)

    def _perturb_rec(self, node: Node) -> Node:
        if node.is_leaf():
            return node
        info = OP_REGISTRY[node.op]
        children = [self._perturb_rec(c) for c in node.children]
        if info["kind"] == "ts":
            lo, hi = info["window_range"]
            key = self._param_key(node)
            new_w = _snap_window(self.perturber.propose(key, node.window, lo, hi), lo, hi)
            return Node(node.op, node.field, new_w, children)
        return Node(node.op, node.field, node.window, children)

    def _param_key(self, ts_node: Node) -> str:
        flds = sorted(ts_node.children[0].fields())
        return f"{ts_node.op}|{flds[0] if flds else '_'}"

    def llm_op(self, tree: Node | None = None) -> Node:
        """LLM 机制引导(阶段 4 provider);阶段 3 stub → 随机。"""
        if self.llm_provider is not None:
            return self.llm_provider(tree, self.fields, self.rng, self._field_usage)
        return random_tree(self.fields, self.cfg.max_depth, self.rng,
                           field_weights=self._field_weights())

    # ---- 批量生成 ----
    def generate(self, parents: list[Node], n: int, llm_time_budget: float = 360.0) -> list[Node]:
        """生成 n 个候选。parents 为空(冷启动)→ 全随机。

        非法结构(arity/窗口)或超深度(LLM 偶发产出 depth>max_depth)的候选【跳过】,
        不足 n 用 random_tree 补足 —— 单个坏候选不崩轮。
        LLM 分支有总时间预算:超预算后降级为随机,避免 API 抖动把生成拖到几十分钟。
        预算 360s = glm-5.3 思考延迟(单次 3-21s × ~15 席)+ 一次挂死(120s 超时)的余量;
        旧值 120s 为 DeepSeek 时代校准,曾致 LLM 分支被单次 hang 耗光预算后静默全降级
        (2026-08-18 轮 370:健康行 生成 ok2 暴露)。
        """
        out: list[Node] = []
        has_parents = len(parents) > 0
        t0 = time.monotonic()
        for _ in range(n * 3):                  # 多试,跳过非法/超深
            if len(out) >= n:
                break
            op = self._pick_op() if has_parents else "random"
            if op == "llm" and time.monotonic() - t0 > llm_time_budget:
                op = "random"                   # LLM 超预算 → 降级随机
            if op == "random":
                cand = random_tree(self.fields, self.cfg.max_depth, self.rng,
                                   field_weights=self._field_weights())
            elif op == "mutate":
                cand = self.mutate(self._pick_parent(parents))
            elif op == "crossover":
                a, b = self._pick_two(parents)
                cand = self.crossover(a, b)
            elif op == "perturb":
                cand = self.perturb_op(self._pick_parent(parents))
            else:  # llm
                cand = self.llm_op(self._pick_parent(parents))
            try:
                cand.validate()
            except Exception:
                continue                        # 非法结构,跳过
            if cand.depth() > self.cfg.max_depth:
                continue                        # LLM/交叉偶发超深,跳过
            out.append(cand)
        while len(out) < n:                     # 仍不足(大量非法)→ random_tree 补足
            c = random_tree(self.fields, self.cfg.max_depth, self.rng,
                            field_weights=self._field_weights())
            c.validate()
            out.append(c)
        return out

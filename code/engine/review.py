# -*- coding: utf-8 -*-
"""M6 审查:五道过滤(前两道简化、后三道拒绝)。

  1. **外层截面算子自动退化**(简化):截面算子(zscore/rank_cs)直接嵌套另一个截面算子时,
     去掉**外层**截面(「外层...退化」),只留内层截面 —— 叠加标准化/排名冗余。
     ※ 指南描述简略,此为实现口径,可调。
  2. **同质算子简化**(拒绝):逐元素 add/sub/mul/div 的两个子树完全相同 → 退化为常数(0/1)、
     冗余(2x)或方向单一(X²,始终非负),拒绝。
  3. **跨量纲运算拒绝**:逐元素 add/sub 两操作数量纲不同 → 拒绝(如 add(close, volume))。
     量纲由 operators.FIELD_DIM + 算子维度规则推导(mul/div 与截面/roc/skew/rank_ts 输出 dimless)。
  4. **最小复杂度门槛**:深度 ≤ 2 直接拒(叶子深度 0)。min_depth 默认 3,可调。
  5. **过度平滑/极值嵌套拒绝**(用户 2026-08-17,研报 §16「隐性过拟合」):
     平滑算子(ma/std)直接嵌套平滑算子(std∘std、ma∘ma、std∘ma…)→ 拒;统计量堆叠使因子
     严重滞后价格变化,能过历史回测但风格切换时反应迟钝。
     极值算子(max/min)直接嵌套极值算子(max∘min、min∘min…)→ 拒;同理(极值嵌套)。

apply(tree) → (简化后的树 | None, 原因)。None 表示被拒。
"""
from __future__ import annotations

from engine.config import REVIEW_MIN_DEPTH as MIN_DEPTH  # [推断] 最小复杂度门槛
from engine.expression import Node
from engine.operators import OP_REGISTRY, field_dimension


def _is_cs(node: Node) -> bool:
    return not node.is_leaf() and OP_REGISTRY[node.op]["kind"] == "cs"


def _flatten_ac(node: Node, op: str) -> list[Node]:
    """收集交换结合算子(add/mul)嵌套树的全部操作数(展平)。"""
    args: list[Node] = []

    def walk(n: Node) -> None:
        if not n.is_leaf() and n.op == op:
            for c in n.children:
                walk(c)
        else:
            args.append(n)

    walk(node)
    return args


def simplify(tree: Node) -> Node:
    """简化与规范化:①同型截面嵌套折叠(zscore(zscore) 等冗余;zscore(rank_cs) **保留**——
    均匀 vs 正态得分形态不同,是生成端可刻意使用的形态,2026-08-18 修正旧口径);
    ②add/mul 交换结合规范化(展平+字典序+左倾重建,等价结构同一 hash)。
    注:zscore(std≈1)与 rank_cs(std≈0.29,比值≈3.4x)的直接混合【保留】(用户 2026-08-18
    拍板:同一量级、合理);真正的量纲/尺度失衡(如 roc(rank_cs) 爆炸支 9x+)由回测前的
    分支支配简化(阈值 SCALE_DOMINANCE=5)处理。返回新树。"""
    if tree.is_leaf():
        return tree
    children = [simplify(c) for c in tree.children]
    node = Node(tree.op, tree.field, tree.window, children)
    # 截面算子的唯一子是【同型】截面 → 去掉外层(冗余);异型(rank_cs/zscore 互包)保留
    if _is_cs(node) and _is_cs(node.children[0]) and node.op == node.children[0].op:
        return node.children[0]
    # add/mul 规范化:展平嵌套同类 → 字典序排序 → 左倾重建
    if node.op in ("add", "mul") and len(node.children) == 2:
        args = _flatten_ac(node, node.op)
        if len(args) > 2:
            args.sort(key=lambda s: s.to_str())
            t = args[0]
            for a in args[1:]:
                t = Node(node.op, None, None, [t, a])
            node = t
    return node


def _out_dim(node: Node) -> str:
    """推断节点输出量纲。叶子用 FIELD_DIM;算子按 dim 规则:same→继承操作数,dimless/ratio→无量纲。"""
    if node.is_leaf():
        return field_dimension(node.field)
    info = OP_REGISTRY[node.op]
    if info["dim"] == "same":
        return _out_dim(node.children[0])
    return info["dim"]  # dimless / ratio


def _degenerate_elem(node: Node) -> bool:
    """add/sub/mul/div 两子树完全相同 → 退化:add=2X(冗余)、sub=0(常数)、mul=X²(方向单一)、div=1(常数)。"""
    if node.is_leaf():
        return False
    if node.op in ("add", "sub", "mul", "div"):
        return node.children[0].to_str() == node.children[1].to_str()
    return False


def _cross_dimension(node: Node) -> bool:
    """add/sub 两操作数量纲不同 → 跨量纲。"""
    if node.is_leaf():
        return False
    if node.op in ("add", "sub"):
        return _out_dim(node.children[0]) != _out_dim(node.children[1])
    return False


_SMOOTH_OPS = {"ma", "std"}    # 平滑统计量
_EXTREME_OPS = {"max", "min"}  # 滚动极值


def _roc_on_cs(node: Node) -> str | None:
    """roc 语义闸(2026-08-18):roc(变化率,做除法)的直接子节点为截面算子(rank_cs/zscore)
    → 拒。rank 有 [0,1] 下界可无限趋零、zscore 会过零——做分母必产生爆炸值
    (实测 roc(rank_cs(·)) 分支截面 std 达 9.6,同库正常分支 0.3)。delta 差分有界,不受限。"""
    for n in node.walk():
        if n.op == "roc" and not n.children[0].is_leaf() \
                and n.children[0].op in ("rank_cs", "zscore"):
            return f"roc_on_cs({n.children[0].op})"
    return None


def _ts_nesting(node: Node) -> tuple[str, str] | None:
    """时序算子直接嵌套同类(过滤5):平滑嵌平滑 / 极值嵌极值。

    只查「直接」嵌套(中间隔截面/逐元素算子不算),研报 §16 口径。
    返回 (类别, outer∘inner);无违规返回 None。
    """
    for n in node.walk():
        if n.is_leaf() or n.window is None:       # 只看时序算子节点
            continue
        child = n.children[0]
        if child.is_leaf() or child.window is None:
            continue
        if n.op in _SMOOTH_OPS and child.op in _SMOOTH_OPS:
            return "oversmoothed", f"{n.op}∘{child.op}"
        if n.op in _EXTREME_OPS and child.op in _EXTREME_OPS:
            return "extreme_nesting", f"{n.op}∘{child.op}"
    return None


def apply(tree: Node, min_depth: int = MIN_DEPTH) -> tuple[Node | None, str]:
    """审查入口:简化 → 复杂度 → 同质退化 → 跨量纲 → 过度平滑/极值嵌套。返回(树|None, 原因)。"""
    t = simplify(tree)
    t.validate()

    if t.depth() < min_depth:
        return None, f"review:min_complexity(depth={t.depth()}<{min_depth})"

    for n in t.walk():
        if _degenerate_elem(n):
            return None, f"review:degenerate_same_subtree({n.op})"
    for n in t.walk():
        if _cross_dimension(n):
            return None, f"review:cross_dimension({n.op}({_out_dim(n.children[0])}≠{_out_dim(n.children[1])}))"
    nest = _ts_nesting(t)
    if nest is not None:
        return None, f"review:{nest[0]}({nest[1]})"
    bad_roc = _roc_on_cs(t)
    if bad_roc:
        return None, f"review:{bad_roc}"

    return t, ""

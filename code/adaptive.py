# -*- coding: utf-8 -*-
"""自适应调度(M3 动态预算)+ 状态信号 —— 让方式 B(Claude /loop)真正「更智能」。

  dynamic_budget(history) → (EvolveConfig, reason):读近 N 轮历史调演化预算。
      · 卡住(连续 0 入库)→ 加探索(随机+LLM)、减利用(M3「高IC候选大量去重→加随机+机制引导」)
      · 有产出 → 偏利用(变异+交叉)(M3「已入库因子附近持续产高分→加变异+扰动」)
      · 历史不足 → 基线
  round_signals(checkpoint) → dict:把检查点浓缩成几条信号,给 Claude 每轮推理/诊断用。

均为 M3 规则的【可测启发式近似】,非精确;真正细粒度的「机制族覆盖/去重命中率」可后续增强。
"""
from __future__ import annotations

from engine.checkpoint import Checkpoint
from engine.evolve import EvolveConfig

_BUDGET_KEYS = ["mutate", "crossover", "perturb", "random", "llm"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def dynamic_budget(history: list[dict], base: EvolveConfig | None = None,
                   recent: int = 5, shift: float = 0.10) -> tuple[EvolveConfig, str]:
    """据近 N 轮历史调整五维预算,返回(新配置, 调整理由)。归一化保证和为 1。"""
    base = base or EvolveConfig()
    if len(history) < 3:
        return base, "baseline(历史<3轮)"
    last = history[-recent:]
    stored = [r.get("n_pass_filters", 0) for r in last]
    avg_stored = _mean(stored)
    stuck = all(s == 0 for s in stored)

    b = {k: getattr(base, k) for k in _BUDGET_KEYS}
    if stuck:
        b["random"] += shift; b["llm"] += shift
        b["mutate"] -= shift; b["perturb"] -= shift
        reason = f"stuck(近{len(last)}轮0入库)→探索+(随机/LLM),利用-(变异/扰动)"
    elif avg_stored > 0:
        s = shift * 0.5
        b["mutate"] += s; b["crossover"] += s
        b["random"] -= s; b["llm"] -= s
        reason = f"yielding(近{len(last)}轮均入库{avg_stored:.1f})→利用+(变异/交叉)"
    else:
        reason = "baseline(有历史但无明显倾向)"
    for k in _BUDGET_KEYS:
        b[k] = max(0.0, b[k])
    tot = sum(b.values())
    cfg = EvolveConfig(**{k: round(b[k] / tot, 4) for k in _BUDGET_KEYS})
    return cfg, reason


def round_signals(checkpoint: Checkpoint) -> dict:
    """浓缩检查点为几条信号,供编排器每轮判断/诊断。"""
    h = checkpoint.history
    recent_stored = [r.get("n_pass_filters", 0) for r in h[-5:]]
    stuck = 0
    for r in reversed(h):
        if r.get("n_pass_filters", 0) == 0:
            stuck += 1
        else:
            break
    top_ic = max((abs(f.get("metrics", {}).get("ic_mean", 0))
                  for f in checkpoint.stored_factors), default=0.0)
    return {
        "iteration": checkpoint.iteration,
        "stored_total": len(checkpoint.stored_factors),
        "tested_total": len(checkpoint.tested_hashes),
        "recent_stored": recent_stored,
        "stuck_rounds": stuck,
        "top_abs_ic": round(top_ic, 4),
    }

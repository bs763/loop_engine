# -*- coding: utf-8 -*-
"""perturb.py 单元测试:梯度方向、冷启动不动、动量平滑、自适应步长、状态持久化。"""
import numpy as np
import pytest

from engine.perturb import Perturber


# ---------------- 梯度方向 ----------------

def test_gradient_positive_when_sharpe_rises_with_n():
    p = Perturber(bandwidth=100)  # 大带宽≈均匀加权,聚焦斜率符号
    for n, s in [(10, 1.0), (20, 2.0), (30, 3.0)]:
        p.observe("ma|close", n, s)
    assert p.gradient("ma|close", 20) > 0


def test_gradient_negative_when_sharpe_falls_with_n():
    p = Perturber(bandwidth=100)
    for n, s in [(10, 3.0), (20, 2.0), (30, 1.0)]:
        p.observe("ma|close", n, s)
    assert p.gradient("ma|close", 20) < 0


def test_gradient_zero_when_insufficient_history():
    p = Perturber()
    p.observe("ma|close", 20, 1.0)  # 仅 1 点
    assert p.gradient("ma|close", 20) == 0.0


# ---------------- 冷启动不动 ----------------

def test_propose_cold_start_noop():
    p = Perturber()
    assert p.propose("ma|close", 20, 3, 250) == 20  # 无历史→原值


# ---------------- 提议方向 ----------------

def test_propose_moves_in_gradient_direction():
    p = Perturber(bandwidth=100, lr=50.0)  # 大 lr 放大步长便于观察方向
    for n, s in [(10, 1.0), (20, 2.0), (30, 3.0)]:
        p.observe("ma|close", n, s)
    assert p.propose("ma|close", 20, 3, 250) > 20  # 正梯度→增大窗口

    p2 = Perturber(bandwidth=100, lr=50.0)
    for n, s in [(10, 3.0), (20, 2.0), (30, 1.0)]:
        p2.observe("ma|close", n, s)
    assert p2.propose("ma|close", 20, 3, 250) < 20  # 负梯度→减小窗口


def test_propose_clipped_to_range():
    p = Perturber(bandwidth=100, lr=1e6)
    for n, s in [(10, 1.0), (20, 2.0), (30, 3.0)]:
        p.observe("ma|close", n, s)
    assert p.propose("ma|close", 20, 3, 250) == 250  # 巨大步长→clip 到 hi


# ---------------- 动量平滑 ----------------

def test_momentum_smooths_sign_flip():
    p = Perturber()
    for _ in range(10):
        p.update_gradient("k", 1.0)      # 稳定正梯度 → m≈1
    assert p.m["k"] > 0.9
    p.update_gradient("k", -1.0)          # 单次反向
    # 动量平滑:m = 0.7*1 + 0.3*(-1) = 0.4 > 0,未完全翻转
    assert 0.3 < p.m["k"] < 0.6


# ---------------- 自适应步长(Adam) ----------------

def test_higher_variance_smaller_step():
    # 两键动量 m 相同(≈2),但梯度波动不同 → v 不同 → 步长不同
    pA = Perturber()
    for _ in range(20):
        pA.update_gradient("k", 2.0)              # 恒定 → v 小
    pB = Perturber()
    for i in range(20):
        pB.update_gradient("k", 1.0 if i % 2 == 0 else 3.0)  # 均值 2、波动大 → v 大

    assert abs(pA.m["k"] - pB.m["k"]) < 0.3       # 动量接近
    assert pB.v["k"] > pA.v["k"]                  # B 波动更大
    # 步长 ∝ m/√v:B 的步长更小
    stepA = abs(pA.lr * pA.m["k"] / (np.sqrt(pA.v["k"]) + 1e-8))
    stepB = abs(pB.lr * pB.m["k"] / (np.sqrt(pB.v["k"]) + 1e-8))
    assert stepB < stepA


# ---------------- 状态持久化 ----------------

def test_state_roundtrip():
    p = Perturber()
    for n, s in [(10, 1.0), (20, 2.0)]:
        p.observe("ma|close", n, s)
    snap = p.state()

    p2 = Perturber()
    p2.load_state(snap)
    assert p2.history == p.history
    assert p2.m == p.m
    assert p2.propose("ma|close", 20, 3, 250) == p.propose("ma|close", 20, 3, 250)

# -*- coding: utf-8 -*-
"""M4 参数扰动:梯度估计 + 动量平滑 + 自适应步长(Adam 风格)。

用于微调时序算子的窗口 n(结构不变,只改窗口)。三步:
  1. **梯度估计(加权最小二乘)**:对同键(算子+字段)的历史 (n, Sharpe) 拟合
     `Sharpe ≈ w0 + w1·n`,斜率 w1 即梯度方向。高斯核加权:距当前 n 越近权重越大(局部邻域)。
     历史不足(< min_history 点)→ 梯度退化为 0,扰动暂停。
  2. **动量平滑(EMA)**:m_t = β·m_{t-1} + (1−β)·g_t,β=0.7。新信息按 30% 融入,避免单次噪声改向。
  3. **自适应步长(Adam)**:step = lr·m / (√v + ε),v = g² 的 EMA。梯度波动大→步长缩小,
     波动小→放大;每参数独立追踪。

key 约定为字符串(如 "ma|close"),由调用方(evolve)按(算子, 主字段)构造。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from engine.config import (  # BETA [研报];LR/BANDWIDTH/MIN_HISTORY [默认·待标定]
    MOMENTUM_BETA as BETA,
    PERTURB_BANDWIDTH as BANDWIDTH,
    PERTURB_LR as LR,
    PERTURB_MIN_HISTORY as MIN_HISTORY,
)


class Perturber:
    """参数扰动器:维护每键的历史/动量/二阶矩,提出新窗口。"""

    def __init__(self, beta: float = BETA, lr: float = LR,
                 bandwidth: float = BANDWIDTH, min_history: int = MIN_HISTORY):
        self.beta = beta
        self.lr = lr
        self.bandwidth = bandwidth
        self.min_history = min_history
        self.history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.m: dict[str, float] = defaultdict(float)   # 动量 EMA
        self.v: dict[str, float] = defaultdict(float)   # 梯度平方 EMA(波动)

    # ---- 梯度估计 ----
    def gradient(self, key: str, center_n: float) -> float:
        """加权最小二乘斜率 w1(高斯核,以 center_n 为中心)。历史不足→0。"""
        pts = self.history.get(key, [])
        if len(pts) < self.min_history:
            return 0.0
        ns = np.array([p[0] for p in pts], dtype=float)
        ss = np.array([p[1] for p in pts], dtype=float)
        w = np.exp(-((ns - center_n) ** 2) / (2 * self.bandwidth ** 2))
        sw = w.sum()
        sn = (w * ns).sum()
        sss = (w * ss).sum()
        snn = (w * ns * ns).sum()
        sns = (w * ns * ss).sum()
        denom = sw * snn - sn * sn
        if abs(denom) < 1e-12:
            return 0.0
        return float((sw * sns - sn * sss) / denom)

    # ---- 状态更新 ----
    def update_gradient(self, key: str, g: float) -> None:
        """用给定梯度 g 更新动量与二阶矩(便于直接测试动量/Adam)。"""
        self.m[key] = self.beta * self.m[key] + (1 - self.beta) * g
        self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (g * g)

    def observe(self, key: str, n: float, sharpe: float) -> None:
        """记录一次 (n, sharpe),并据此更新该键的动量/二阶矩。"""
        self.history[key].append((float(n), float(sharpe)))
        self.update_gradient(key, self.gradient(key, float(n)))

    # ---- 提议新窗口 ----
    def propose(self, key: str, current_n: int, lo: int, hi: int) -> int:
        """提出新窗口:冷启动(历史不足)→ 原值;否则 current_n + Adam 步长,clip 到 [lo,hi]。"""
        if len(self.history.get(key, [])) < self.min_history:
            return int(current_n)
        m = self.m[key]
        v = self.v[key]
        step = self.lr * m / (np.sqrt(v) + 1e-8)
        new_n = int(round(current_n + step))
        return max(lo, min(hi, new_n))

    # ---- 持久化 ----
    def state(self) -> dict:
        return {
            "history": {k: v for k, v in self.history.items()},
            "m": dict(self.m),
            "v": dict(self.v),
            "beta": self.beta, "lr": self.lr, "bandwidth": self.bandwidth,
        }

    def load_state(self, s: dict) -> None:
        self.history = defaultdict(list, {k: [tuple(p) for p in v]
                                          for k, v in s.get("history", {}).items()})
        self.m = defaultdict(float, s.get("m", {}))
        self.v = defaultdict(float, s.get("v", {}))

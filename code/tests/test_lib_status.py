# -*- coding: utf-8 -*-
"""lib_status 双口径相关性体检测试:违规/灰区分类,PnL 口径捕捉 IC 口径盲区。"""
import numpy as np

from lib_status import corr_report_lines


def _factor(expr, ic, ret):
    return {"expr": expr, "ic_series": ic, "ls_ret": ret}


def test_dual_gauge_gray_and_violation():
    rng = np.random.default_rng(0)
    n = 300
    base_ret = rng.normal(0, 0.01, n)
    # A/B:IC 不相关但日收益高度同步(共同暴露)→ PnL 口径违规、IC 口径干净
    # A/C:IC 中度相关、收益中度相关 → 双口径灰区
    f = [
        _factor("fA", rng.normal(0, 1, n), base_ret + rng.normal(0, 0.002, n)),
        _factor("fB", rng.normal(0, 1, n), 0.95 * base_ret + rng.normal(0, 0.004, n)),
        _factor("fC", 0.55 * np.asarray(f := rng.normal(0, 1, n)) + rng.normal(0, 0.8, n),
                0.55 * base_ret + rng.normal(0, 0.008, n)),
    ]
    # 修正 fC 构造(上面 walrus 写法只影响 ic)——直接重算
    ic_a = f[0]["ic_series"]
    f[2]["ic_series"] = 0.55 * ic_a + rng.normal(0, 0.8, n)
    lines = corr_report_lines(f, show_gray=10)
    text = "\n".join(lines)
    ic_line = next(l for l in lines if "IC口径" in l)
    pnl_line = next(l for l in lines if "PnL口径" in l)
    assert "3 对" in ic_line and "3 对" in pnl_line
    # PnL 口径:fA×fB ≥0.7 违规;fA×fC、fB×fC 依构造应落在灰区或违规
    assert "≥0.7 共 1 对" in pnl_line, pnl_line
    assert "灰区" in pnl_line
    # IC 口径:fA×fB 不违规(IC 独立),fA×fC 灰区(0.55 相关)
    assert "≥0.7 共 0 对" in ic_line, ic_line


def test_missing_series_graceful():
    lines = corr_report_lines([{"expr": "onlyIc", "ic_series": [0.1] * 30}])
    assert any("样本不足" in l for l in lines)   # PnL 口径无 ls_ret → 提示而非崩

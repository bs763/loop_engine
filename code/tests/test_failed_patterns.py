# -*- coding: utf-8 -*-
"""failed_patterns.py 单元测试:内因判定(占位 vs 死证据)、判死、prompt 注入、原子写、回填口径。"""
import json

from engine.expression import parse
from engine import failed_patterns as F


def _setup(tmp_path):
    """隔离:换路径 + 清缓存;返回恢复函数。"""
    old_path = F._PATH
    F._reset(path=tmp_path / "failed_patterns.json", lib={})

    def _restore():
        F._reset(path=old_path, lib={})
    return _restore


def test_pure_occupied_rejection_counts_occupied_not_fails(tmp_path):
    """纯 #9/#15 拒(占位灭)→ occupied,不算死证据,不判死。"""
    _r = _setup(tmp_path)
    try:
        node = parse("add(rank_cs(close), rank_cs(volume))")
        for _ in range(30):
            F.record_reject(node, "filter_reject",
                            ["9.IC相关性=0.833≥0.7"], 5)
        e = F.failed_patterns()["add(rank_cs(FLD), rank_cs(FLD))"]
        assert e["occupied"] == 30 and e["fails"] == 0
        assert F.dead_skeletons() == set()
    finally:
        _r()


def test_intrinsic_rejections_make_skeleton_dead(tmp_path):
    """内因拒(规则 1/13 等)→ fails;≥10 且 0 成功 → 判死并注入 prompt。"""
    _r = _setup(tmp_path)
    try:
        node = parse("add(rank_cs(close), rank_cs(volume))")
        for _ in range(10):
            F.record_reject(node, "filter_reject",
                            ["1.|IC|=0.0079≤0.03", "13.单调性=0.176≤0.85"], 5)
        assert "add(rank_cs(FLD), rank_cs(FLD))" in F.dead_skeletons()
        blk = F.prompt_block()
        assert "add(rank_cs(FLD), rank_cs(FLD))" in blk and "×10连败" in blk
        assert "IC弱" in blk or "单调性低" in blk   # 头号内因类别
        assert "0.0079" not in blk                  # 指标数值不回流
    finally:
        _r()


def test_mixed_reasons_count_as_intrinsic(tmp_path):
    """#9 与内因规则并存 → 内因灭(其余规则独立足以拒),#9 不进 top_rules。"""
    _r = _setup(tmp_path)
    try:
        node = parse("skew(close, 40)")
        F.record_reject(node, "filter_reject",
                        ["9.IC相关性=0.8≥0.7", "1.|IC|=0.01≤0.03"], 5)
        e = F.failed_patterns()["skew(FLD, N)"]
        assert e["fails"] == 1 and e["occupied"] == 0 and "9" not in e["top_rules"]
    finally:
        _r()


def test_review_and_backtest_error_are_intrinsic(tmp_path):
    """review 结构拒 / 回测异常 → 直接死证据。"""
    _r = _setup(tmp_path)
    try:
        F.record_reject(parse("ma(close, 20)"), "review_reject",
                        ["min_complexity depth=1<2"], 5)
        F.record_reject(parse("std(close, 20)"), "backtest_error",
                        ["ValueError: 覆盖率塌陷"], 5)
        lib = F.failed_patterns()
        assert lib["ma(FLD, N)"]["top_rules"] == {"review": 1}
        assert lib["std(FLD, N)"]["top_rules"] == {"bt_err": 1}
    finally:
        _r()


def test_stored_history_prevents_dead(tmp_path):
    """同骨架曾有成功 → 不判死(被替换出库也不扣减)。"""
    _r = _setup(tmp_path)
    try:
        node = parse("add(rank_cs(close), rank_cs(volume))")
        F.record_stored(node, 3)
        for _ in range(20):
            F.record_reject(node, "filter_reject", ["1.|IC|=0.01≤0.03"], 5)
        assert F.dead_skeletons() == set()
    finally:
        _r()


def test_save_atomic_and_reloadable(tmp_path):
    """轮末落盘(原子写)后重载,数据完整、updated_iter 正确。"""
    _r = _setup(tmp_path)
    try:
        F.record_reject(parse("skew(close, 40)"), "review_reject", ["x"], 7)
        F.save(7)
        data = json.loads((tmp_path / "failed_patterns.json").read_text(encoding="utf-8"))
        assert data["updated_iter"] == 7
        assert data["patterns"]["skew(FLD, N)"]["fails"] == 1
        assert not (tmp_path / "failed_patterns.json.tmp").exists()
        F._reset(lib=None)   # 清缓存重读
        assert F.failed_patterns()["skew(FLD, N)"]["fails"] == 1
    finally:
        _r()


def test_generation_prompt_injects_dead_and_not_occupied(tmp_path):
    """生成 prompt:死骨架注入;纯占位骨架绝不注入(保优淘劣挑战者路径)。"""
    _r = _setup(tmp_path)
    try:
        dead = parse("add(rank_cs(close), rank_cs(volume))")
        occ = parse("zscore(std(rank_cs(close), 20))")
        for _ in range(12):
            F.record_reject(dead, "filter_reject", ["13.单调性=0.5≤0.85"], 5)
        for _ in range(50):
            F.record_reject(occ, "filter_reject", ["9.IC相关性=0.9≥0.7"], 5)
        F.save(5)
        from llm import mechanisms as M
        mech = M.MECHANISMS[0]
        prompt = M.build_generation_prompt(mech, ["close", "volume"])
        assert "全灭结构" in prompt and "add(rank_cs(FLD), rank_cs(FLD))" in prompt
        assert "zscore(std(rank_cs(FLD), N))" not in prompt
    finally:
        _r()

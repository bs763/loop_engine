# -*- coding: utf-8 -*-
"""checkpoint.py 单元测试:原子写、恢复一致、去重、空检查点、子模块状态。"""
import json
from pathlib import Path

from engine.checkpoint import Checkpoint


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "cp.json"
    cp = Checkpoint(p)
    cp.iteration = 42
    cp.add_tested("abc123")
    cp.add_tested("def456")
    cp.add_factor({"expr": "zscore(ma(close, 20))", "ic_mean": 0.05, "ic_series": [0.04, 0.05, 0.06]})
    cp.perturb_state = {"m": {"ma|close": 0.3}, "v": {"ma|close": 0.1}}
    cp.fsa_state = {"counts": {"ma(FLD, N)": 3}}
    cp.save()

    loaded = Checkpoint.load(p)
    assert loaded.iteration == 42
    assert loaded.tested_hashes == {"abc123", "def456"}
    assert loaded.stored_factors == cp.stored_factors
    assert loaded.perturb_state == cp.perturb_state
    assert loaded.fsa_state == cp.fsa_state


def test_atomic_no_tmp_leftover(tmp_path):
    p = tmp_path / "cp.json"
    cp = Checkpoint(p)
    cp.iteration = 1
    cp.save()
    # 落盘后:目标文件存在、是合法 JSON、无 .tmp 残留
    assert p.exists()
    json.load(open(p, encoding="utf-8"))  # 不抛异常即合法
    assert not Path(str(p) + ".tmp").exists()


def test_load_missing_returns_empty(tmp_path):
    cp = Checkpoint.load(tmp_path / "nope.json")
    assert cp.iteration == 0
    assert cp.tested_hashes == set()
    assert cp.stored_factors == []


def test_dedup_via_tested_hashes(tmp_path):
    cp = Checkpoint(tmp_path / "cp.json")
    assert not cp.is_tested("h1")
    cp.add_tested("h1")
    assert cp.is_tested("h1")
    cp.add_tested("h1")  # 重复不增
    assert len(cp.tested_hashes) == 1


def test_overwrite_preserves_latest(tmp_path):
    p = tmp_path / "cp.json"
    cp = Checkpoint(p)
    cp.iteration = 1
    cp.save()
    cp.iteration = 99
    cp.add_tested("zzz")
    cp.save()  # 覆盖写
    loaded = Checkpoint.load(p)
    assert loaded.iteration == 99
    assert "zzz" in loaded.tested_hashes

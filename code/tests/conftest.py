# -*- coding: utf-8 -*-
"""测试全局隔离:失败模式库与终审审计日志重定向到 tmp,防 run_round 污染真 output。"""
import pytest

from engine import failed_patterns as fplib


@pytest.fixture(autouse=True)
def _isolate_runtime_outputs(tmp_path):
    old = fplib._PATH
    fplib._reset(path=tmp_path / "failed_patterns.json", lib={})
    import loop_orchestrate as _lo
    old_out = _lo.OUTPUT_DIR
    _lo.OUTPUT_DIR = tmp_path          # 终审审计 final_review_log.jsonl 等
    yield
    fplib._reset(path=old, lib={})
    _lo.OUTPUT_DIR = old_out

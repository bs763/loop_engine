# -*- coding: utf-8 -*-
"""测试全局隔离:失败模式库重定向到 tmp,防 run_round(fplib.save)污染真库。"""
import pytest

from engine import failed_patterns as fplib


@pytest.fixture(autouse=True)
def _isolate_failed_patterns(tmp_path):
    old = fplib._PATH
    fplib._reset(path=tmp_path / "failed_patterns.json", lib={})
    yield
    fplib._reset(path=old, lib={})

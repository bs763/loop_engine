# -*- coding: utf-8 -*-
"""框架冒烟测试:确认 pytest + 扁平包导入链路通。"""


def test_imports():
    from data_layer import load_factor_data  # noqa: F401
    assert True


def test_arithmetic():
    assert 1 + 1 == 2

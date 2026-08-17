# -*- coding: utf-8 -*-
"""数据层:OSS 取数 + 本地缓存 + 前复权 + 派生字段计算。

主要入口:
  - load_factor_data(start_year, end_year) → 含全部原始+派生字段的最终 DataFrame
  - build_factor_table(...)                → base 表(前复权+对齐,未算派生)
  - compute_derived(df)                    → 在 base 表上追加 10 派生字段
  - oss.connect()                          → 配好 OSS 内网凭证的 DuckDB 连接

文件:oss.py(连接+路径)、data_loader.py(缓存+前复权+对齐)、derived_fields.py(派生)。
"""
from data_layer import oss  # noqa: F401
from data_layer.data_loader import build_factor_table, load_factor_data  # noqa: F401
from data_layer.derived_fields import compute_derived, DERIVED_FIELDS  # noqa: F401

__all__ = [
    "oss",
    "build_factor_table",
    "load_factor_data",
    "compute_derived",
    "DERIVED_FIELDS",
]

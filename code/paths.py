# -*- coding: utf-8 -*-
"""项目路径集中管理。

所有脚本统一从这里取 cache/ 与 output/ 的绝对路径,避免各处内联相对路径 hack。
路径基准是本文件自身位置(code/paths.py),与「从哪里启动 / cwd 是什么」无关,稳定可靠。

目录约定(见 docs/项目执行指南.md 与 factor_loop_engine/README.md):
  cache/   OSS 拉取的可重建缓存(行情/复权/股本 parquet)—— 可删重拉,建议 gitignore
  output/  最终产出(阶段报告 / 因子库 / checkpoint / 结果)
"""
from __future__ import annotations

from pathlib import Path

# code/ 的父目录 = 项目根 factor_loop_engine/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# OSS 缓存(可重建中间数据)
CACHE_DIR: Path = PROJECT_ROOT / "cache"
# 最终产出(交付物)
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
# 文档(只读规格)
DOCS_DIR: Path = PROJECT_ROOT / "docs"


def ensure_dirs() -> None:
    """启动时确保 cache/ output/ 存在(幂等)。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

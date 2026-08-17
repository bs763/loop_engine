# -*- coding: utf-8 -*-
"""alphalab 真实回测适配器:因子面板 → alphalab check → FactorMetrics。

调用用户的 alphalab(在 因子检测操作步骤/.venv):
    alphalab check <因子.parquet> -c my.yaml -d <结果目录> [-n 名称] [--gate]
输入因子 parquet:宽表,date 列 + 每只股票(order_book_id)一列,float 值(与示例一致)。
结果目录解析:overview_ic.csv / overview_group.csv / h{H}_stats.json /
            group/h{H}_ls_by_year.csv / ic/h{H}_by_year.csv / ic/h{H}_ic_series.parquet /
            group/h{H}_long_excess_nav.parquet → FactorMetrics(horizon 默认 5)。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.interface import Evaluator, FactorMetrics
from paths import CACHE_DIR

# 本机默认路径;换电脑用环境变量 ALPHALAB_DIR 覆盖(见 .env / .env.example)
FALLBACK_ALPHALAB_DIR = r"C:\Users\Administrator\Desktop\因子检测操作步骤"


class AlphalabEvaluator(Evaluator):
    def __init__(self, alphalab_dir: str | Path | None = None,
                 config_yaml: str | Path | None = None, horizon: int = 5,
                 out_root: str | Path | None = None,
                 in_root: str | Path | None = None,
                 gate: bool = False, timeout: int = 600,
                 keep_output: bool = False):
        self.alphalab_dir = Path(alphalab_dir) if alphalab_dir else Path(
            os.environ.get("ALPHALAB_DIR", FALLBACK_ALPHALAB_DIR))
        self.config_yaml = Path(config_yaml) if config_yaml else self.alphalab_dir / "my.yaml"
        self.horizon = horizon
        self.out_root = Path(out_root) if out_root else (CACHE_DIR / "alphalab_out")
        self.in_root = Path(in_root) if in_root else (CACHE_DIR / "alphalab_in")
        self.gate = gate
        self.timeout = timeout
        self.keep_output = keep_output   # True=保留回测结果目录(默认 False:解析完即删,省盘)

    # ---- 可执行命令 ----
    def _exe_cmd(self) -> list[str]:
        exe = self.alphalab_dir / ".venv" / "Scripts" / "alphalab.exe"
        if exe.exists():
            return [str(exe)]
        py = self.alphalab_dir / ".venv" / "Scripts" / "python.exe"
        return [str(py), "-m", "alphalab"]

    # ---- 写因子 parquet(alphalab 输入格式)----
    def write_panel(self, panel: pd.DataFrame, name: str) -> Path:
        from engine.config import BACKTEST_END, BACKTEST_START
        df = panel.copy()
        df.index.name = "date"
        df.columns = df.columns.astype(str)
        df = df.astype("float32")
        df = df.loc[BACKTEST_START:BACKTEST_END]   # 只回测 2018-2025(warmup 部分不回测)
        self.in_root.mkdir(parents=True, exist_ok=True)
        path = self.in_root / f"{name}.parquet"
        df.to_parquet(path)  # index.name='date' → 写出 'date' 列 + 股票列
        return path

    # ---- 跑 alphalab ----
    def _run(self, parquet: Path, outdir: Path, name: str) -> subprocess.CompletedProcess:
        cmd = self._exe_cmd() + ["check", str(parquet), "-c", str(self.config_yaml),
                                 "-d", str(outdir), "-n", name]
        if self.gate:
            cmd.append("--gate")
        return subprocess.run(cmd, cwd=str(self.alphalab_dir), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=self.timeout)

    def evaluate(self, panel: pd.DataFrame, name: str = "factor") -> FactorMetrics:
        # 确定性坏因子拦截:几乎全 NaN/Inf 的因子 alphalab 必拒(要求每交易日覆盖),提前拦下省一次回测。
        # 阈值 1% 极保守(正常因子有限值占比 >50%),零误伤。
        arr = np.asarray(panel.values, dtype=float)
        if arr.size and float(np.isfinite(arr).mean()) < 0.01:
            raise ValueError(f"因子几乎全非有限(finite={np.isfinite(arr).mean():.4f}<0.01)")
        in_path = self.write_panel(panel, name)
        outdir = self.out_root / name
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            proc = self._run(in_path, outdir, name)
            if not (outdir / "overview_ic.csv").exists():
                raise RuntimeError(
                    f"alphalab 未产出结果(returncode={proc.returncode})\n"
                    f"stderr:\n{proc.stderr}")
            m = self.parse_dir(outdir, horizon=self.horizon, name=name)
            if self.gate:
                m.admission_pass = (proc.returncode == 0)
            return m
        finally:
            # 解析完即删结果目录+输入 parquet(指标已进 FactorMetrics;默认 keep_output=False 省盘)
            # 用 finally 保证 alphalab 异常/超时/解析失败也清理,避免失败候选漏删爆盘
            if not self.keep_output:
                shutil.rmtree(outdir, ignore_errors=True)
                in_path.unlink(missing_ok=True)

    # ---- 解析结果目录(可独立调用,便于离线测试)----
    @staticmethod
    def parse_dir(outdir: str | Path, horizon: int = 5,
                  name: str = "factor", expr: str = "") -> FactorMetrics:
        outdir = Path(outdir)
        H = horizon
        ic = pd.read_csv(outdir / "overview_ic.csv")
        grp = pd.read_csv(outdir / "overview_group.csv")
        meta = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))
        ic_row = ic[ic["预测期"] == H].iloc[0]
        grp_row = grp[grp["预测期"] == H].iloc[0]

        # 按年(alphalab 0.4.2 起年份列改为无名索引列,index_col=0 读成年)
        ls_year = pd.read_csv(outdir / "group" / f"h{H}_ls_by_year.csv", index_col=0)
        ic_year = pd.read_csv(outdir / "ic" / f"h{H}_by_year.csv", index_col=0)
        annual_ls = {int(idx): float(r["return"]) for idx, r in ls_year.iterrows()}
        annual_ic = {int(idx): float(r["ic_mean"]) for idx, r in ic_year.iterrows()}

        # 序列
        ic_ser = pd.read_parquet(outdir / "ic" / f"h{H}_ic_series.parquet")
        le = pd.read_parquet(outdir / "group" / f"h{H}_long_excess_nav.parquet")
        ls_nav_df = pd.read_parquet(outdir / "group" / f"h{H}_ls_nav.parquet")

        # 多头超额夏普(h{H}_stats.json)
        le_sharpe = float("nan")
        stats_path = outdir / "group" / f"h{H}_stats.json"
        if stats_path.exists():
            st = json.loads(stats_path.read_text(encoding="utf-8"))
            le_sharpe = float(st.get("long_excess_stats", {}).get("sharpe", float("nan")))

        ls_annual = float(grp_row["ls_annual"])
        ls_max_dd = float(grp_row["ls_max_dd"])
        calmar = ls_annual / abs(ls_max_dd) if abs(ls_max_dd) > 1e-12 else float("nan")
        monotonicity = float(grp_row["monotonicity"])

        return FactorMetrics(
            expr=expr,
            direction=int(meta.get("direction", 0)),
            horizon=H,
            ic_mean=float(ic_row["ic_mean"]), icir=float(ic_row["icir"]),
            icir_annual=float(ic_row["icir_annual"]), t_stat_nw=float(ic_row["t_stat_nw"]),
            positive_ratio=float(ic_row["positive_ratio"]),
            ls_annual=ls_annual, ls_sharpe=float(grp_row["ls_sharpe"]),
            ls_max_dd=ls_max_dd, calmar=calmar, monotonicity=monotonicity,
            long_excess_annual=float(grp_row["long_excess_annual"]), long_excess_sharpe=le_sharpe,
            annual_ls_return=annual_ls, annual_ic=annual_ic,
            ic_series=[float(x) for x in ic_ser["ic"].tolist()],
            long_excess_nav=[float(x) for x in le["long_excess"].tolist()],
            ls_nav=[float(x) for x in ls_nav_df["net"].tolist()],
            meta={"name": name, "alphalab_version": meta.get("alphalab_version"),
                  "sample": meta.get("sample")},
        )

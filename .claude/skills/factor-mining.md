---
name: factor-mining
description: 自动化因子发现引擎(factor_loop_engine)。需要跑因子挖掘轮次、查看因子库、或调试生成/审查/回测流程时使用。每轮自动加载项目背景与运行方式,避免重复推导。
---

# 自动化因子发现引擎 · Skill

`factor_loop_engine/` 实现中金《Loop+Engineering 自动化因子发现引擎》。核心算法层纯 Python,
Claude Code(`/loop`/cron)只做调度。详见 `docs/项目执行指南.md` 与 `output/claude_code_integration.md`。

## 一轮怎么跑(在 `factor_loop_engine/` 下)
- 离线 mock(无 LLM/无 alphalab):`uv run --directory factor_loop_engine code/run_round_cli.py --mock --n 100`
- 真实(DeepSeek 生成 + alphalab 回测):`uv run --directory factor_loop_engine code/run_round_cli.py --n 100`
- 查因子库状态:`uv run --directory factor_loop_engine code/lib_status.py`
- 断点续跑:检查点 `output/checkpoint.json` 自动加载/原子保存。清零就删它。

## 关键事实(勿每轮重新推导)
- **数据**:阿里云 OSS `ys-datalake` + duckdb;日频前复权;overnight 用 `shift(adj_close)`(非 prev_close);
  mv=adj_close×free_circulation。字段宽表缓存 `cache/panels/`。
- **算子**:14 种(时序8/逐元素4/截面2);表达式深度 ≤ 4。
- **演化**:预算 25/25/15/15/20(变异/交叉/扰动/随机/LLM);动量 β=0.7。
- **FSA**:支持度 > 15% 且 ≥ 2 次冻结;同骨架变体上限 5。
- **回测**:alphalab(universe=中证全指、行业+市值中性、单边千一、horizon=5、2018-2025)。
- **14 项过滤**(多空 ls 为超额口径):`|IC|>0.03` / 每年 ls>0 / 整体 ls 年化>0 / 夏普>0.5 /
  末年夏普>0.5 / Calmar>1.0 / 近9·12月 ls>0 / IC 相关性<0.70 / FSA / 失败库 /
  **多头超额年化>0** / **单调性>0.85** / **ICIR>0.3**(末三条为用户加严)。
- **LLM**:生成+审查默认 DeepSeek(`.env` 配,可切 GLM)。
- **防 Goodhart**:生成/审查只处理表达式,回测指标在后半段产生且**不回流**给生成端。
- **参数出处**全在 `code/engine/config.py`(标 [研报]/[推断]/[默认])。

## 目录
`code/` 代码 · `cache/` 可重建缓存 · `output/` 产出(报告+检查点)· `docs/` 规格 · `tests/` 128 测试。

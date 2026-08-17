---
name: factor-loop-orchestrator
description: factor_loop_engine 的连续调度编排器例程。链式触发(完成上一轮立刻启动下一轮)时按此跑一轮因子挖掘:读状态→跑(脚本自带动态预算)→诊断→记教训→一句话汇报。让 Claude 的自适应智能真正介入。
---

# 因子挖掘 /loop 编排器例程(方式 B)

每轮(连续链式触发:完成上一轮即启动本轮)按此执行。**目标:让 Claude 的判断介入调度,而非只按按钮。**
项目背景见 `factor-mining.md` 技能;对接全貌见 `output/claude_code_integration.md`。

## 每轮六步(保持简短,省 token)

1. **读状态**:`uv run --directory factor_loop_engine code/lib_status.py`;再看上轮 `output/loop.log` 末尾的 `SIGNALS:` 行(stuck_rounds / recent_stored / top_abs_ic)。
2. **跑一轮**(脚本已自带 M3 动态预算,无需你设):
   `uv run --directory factor_loop_engine code/run_round_cli.py --n 100`(真实模式)
   或 `--mock`(离线)。
3. **读产出**:看本轮 `STATUS:` 与 `SIGNALS:` 行。
4. **诊断**(Claude 的价值所在):
   - 长期 0 入库( stuck_rounds 大)且 top_abs_ic 很低 → 多半字段/前复权/口径有问题,去查 `output/loop.log` 的 reject 样本;**别盲目多跑**。
   - 回测反复异常(alphalab 报错)→ 查 rqdatac 缓存 / 输入 parquet 格式。
   - 某 reject 原因反复出现 → 记教训(第 5 步)。
5. **记教训**(自我改进):确认是可复用经验后,往 `output/lessons.md` 追加一行:`- [日期] [现象] → [处理]`。**只记确实复用的,别灌水。**
6. **一句话汇报**:已测/入库/Top IC + 是否记了新教训。**不要展开日志。**

## 边界(三道防线)
- 连续 N 轮(建议 ≥8)无新因子且诊断无果 → 停下来报告,别空转烧 token。
- 高风险动作(删 checkpoint、改 .env、动 alphalab 配置)→ 先问用户。
- 全程不过夜狂跑:注意 DeepSeek 余额与 alphalab rqdatac 并发。

## 切回最省模式
若只是想无人值守纯挖因子(不需自适应),改用 OS cron 直跑 `run_round_cli.py`(方式 A,零调度 token),见对接指南 §2-A。

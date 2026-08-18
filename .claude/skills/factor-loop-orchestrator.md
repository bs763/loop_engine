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
6. **每轮汇报——完整格式,逐项必填**(用户 2026-08-18 要求,不许压缩成一句话):
   ```
   ═══ 轮 N ═══
   模式: 探索/深挖/基线 + 原因与预算(原样引 BUDGET 行,如
        「深挖 yielding(近5轮均入库1.0)→利用+ | 0.3/0.3/0.15/0.1/0.15」)
   漏斗: 生成X→新测Y→审查过Z→回测Z→入库W(累计M) [耗时] | TopIC | stuck | tested_total
   LLM: 生成 ok../解析失败../API错../兜底.. | 终审 ok../拒../错..
   OOS体检: (原样引用体检行)
   入库/替换明细:(入库=0 也写"本轮无入库")
     每个新因子: 表达式 + IS指标(IC/夏普/ICIR/多头超额) + OOS指标 + 替换对象
     (旧因子 hash/表达式/综合分对比,注明触发路径 #9 相关 或 #15 同构)
   触发项(逐项报,无则标"无"):
     · 相关性体检: 两两N对,≥0.7 共X对(违规列出)
     · 回测异常: 准确数字(取自 REJECT分类 的「回测异常×N」;分类行无此项 = 0;
       基线0-4/轮,≥10 或连续两轮偏高才展开诊断)
     · 新闸拦截: 覆盖率/过度平滑/极值嵌套 各N(异常放量才展开)
     · 族记忆: family_notes 新增条目(原文)或"无新增"
     · lessons: 新条目或"无"
   ```
   模式取自 BUDGET 行【】标签(探索=连续5轮0入库加随机+LLM;深挖=持续产出加变异+交叉;基线=无倾向)。
   数据来源:loop.log 末尾各行 + checkpoint 最新因子 + git 上一版 checkpoint(被替换者) + lib_status(相关/OOS)。
   - 入库/替换必带表达式明细(新因子含 OOS 存档值;替换注明几进几出)
   - 有则附:相关性体检 ≥0.7 对、回测异常 ≥10、新闸(覆盖率/过度平滑/极值嵌套)异常放量、
     族记忆新增(family_notes,终审拒因回流)、lessons 新条目
   - 停机即报(四道防线):**OOS ALERT**(中位 OOS IC≤0 或负占比≥50%,立即停+汇报全库衰减明细)、
     stuck ≥8 且无果、高风险动作先问、每晚 ≥21:00 首轮确认 git 备份推送结果

## 边界(四道防线)
- 连续 N 轮(建议 ≥8)无新因子且诊断无果 → 停下来报告,别空转烧 token。
- 高风险动作(删 checkpoint、改 .env、动 alphalab 配置)→ 先问用户。
- 全程不过夜狂跑:注意 GLM 配额与 alphalab rqdatac 并发。
- **OOS ALERT(用户 2026-08-18 拍板)**:每轮 STATUS 后的「OOS体检」行出现 ALERT
  (中位 OOS IC≤0 或负占比≥50%)→ **立即停止续链**,向用户汇报库内 IS→OOS 衰减明细,
  等待人工决策(OOS 是诊断信号,绝不程序化用它筛因子)。

## 每日异地备份(2026-08-17 增,远程 github.com/bs763/loop_engine)
夜间窗口(≥21:00)当天的**第一轮**完成后,做一次备份提交并推送:

```
git add -A; git add -f output/checkpoint.json
git commit -m "夜间备份: iter=<轮次> stored=<库存数>(自动)"
git push          # 凭据已存 Windows 凭据管理器,免交互
```

git 不在 PATH 时用完整路径 `& "C:\Program Files\Git\cmd\git.exe" <子命令>`。
push 失败(网络/权限)不阻塞跑轮,下一轮再试;连续失败在当日汇报里提一句。
当天已备份过则跳过(看 `git log -1 --format=%cd` 是否今天)。

## 切回最省模式
若只是想无人值守纯挖因子(不需自适应),改用 OS cron 直跑 `run_round_cli.py`(方式 A,零调度 token),见对接指南 §2-A。

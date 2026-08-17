# Claude Code 对接指南 —— 把核心算法层组装成定时闭环

> 阶段 7 产出。核心算法层(阶段 1–6,128 测试)已是纯 Python、mock 端到端跑通;本文说明如何把它跑成
> **定时调度、断点续跑、(可选)无人值守**的因子挖掘闭环,以及三种启动方式的成本取舍。
> 适用对象:在 `factor_loop_engine/` 目录用 Claude Code(或 OS 调度)运行。

---

## 0. 一句话现状

单轮全流程已封装进 `code/run_round_cli.py`:**生成 → 审查 → 回测 → 十一项过滤 → 入库 → 检查点**。
它能被任何调度器(OS cron / Claude `/loop`)反复调用;检查点 `output/checkpoint.json` 保证断点续跑。
**Claude 不是必需的**——脚本自身就能自动跑轮次;Claude 只在「需要自适应调控 / 人介入」时才进循环。

---

## 1. 架构总览

```
        ┌──────────────── 调度层(三选一,见 §2)────────────────┐
        │  A. OS cron(Task Scheduler)   ← 最省,Claude 不参与    │
        │  B. Claude 连续调度(链式)   ← Claude 每轮介入        │
        │  C. 物理隔离 sub-agent        ← 最强 anti-Goodhart      │
        └────────────────────────┬──────────────────────────────┘
                                 ▼ 每次调用
   ┌─────────────────── 核心算法层(纯 Python,loop_orchestrate.py)──────────────────┐
   │ 生成(Evolver)→ 审查(review 四过滤)→ 去重 → 回测(Evaluator)→ 14 过滤 → 入库   │
   └────────────────────────────┬───────────────────────────────────┬──────────────┘
                                 ▼                                   ▼
            LLM 子代理层(DeepSeek 生成+审查)                  真实回测(alphalab)
            llm/settings.py + mechanisms.py                     backtest/alphalab_adapter.py
```

**防 Goodhart(指南 §2)**:`loop_orchestrate.py` 内已做**逻辑隔离**——生成/审查函数只处理表达式,
回测指标在后半段产生且**不传入生成端**(代码层面强制,生成函数签名里就没有指标)。
物理隔离(独立子代理 + 临时 JSON)是可选强化(见 §2-C)。

---

## 2. 三种启动方式(按 token 成本从低到高)

> 关键认知(用户 2026-08-11 拍板):**调度层才是 token 大头**(研报:每轮 43K 中调度占 35K=80%)。
> Python 里的 DeepSeek 调用(生成+审查)只占 ~20%。所以「省 token」= **让 Claude 尽量不进循环**。

### A. OS 级 cron —— 最省(推荐长期生产用)

OS 定时器直接跑脚本,Claude 完全不参与 → **零调度 token**。DeepSeek/alphalab 在脚本内跑。

**Windows(Task Scheduler)**,建一个 `factor_loop_engine/run_round.bat`:
```bat
@echo off
cd /d C:\Users\Administrator\Desktop\codewhale\factor_loop_engine
uv run --directory . code\run_round_cli.py --n 100 >> output\loop.log 2>&1
```
注册每 30 分钟跑一次:
```powershell
schtasks /Create /SC MINUTE /MO 30 /TN "FactorMining" /TR "C:\Users\Administrator\Desktop\codewhale\factor_loop_engine\run_round.bat" /F
```
查状态:`uv run --directory factor_loop_engine code\lib_status.py`(或看 `output/loop.log` 末尾 STATUS 行)。

**Linux**:`crontab -e` → `*/30 * * * * cd .../factor_loop_engine && uv run --directory . code/run_round_cli.py --n 100 >> output/loop.log 2>&1`

> 适合:纯自动挖因子(脚本已能自走生成→审查→回测→入库)。Claude 只在你想看结果/调方向时手动介入。

### B. Claude 连续调度 —— Claude 每轮介入(当前在用)

**连续模式**:一轮跑完(后台任务完成)→ Claude 立刻启动下一轮 → 完了再启动 → ... 链式连续,
**无固定间隔、零等待**。比固定 cron 省掉每轮空闲等待,吞吐提升 ~1.7×。

机制:
- 一轮完成(task 通知)→ Claude 读 STATUS/SIGNALS → 一句话汇报 → **立刻后台启动下一轮** → ...
- 不用 CronCreate(无固定时间触发);链由「完成→启动」维持。
- Claude 仍在循环里(方式 B):每轮诊断异常、记教训、自适应预算(脚本内自动)。

- **成本**:每轮 = Claude 一次编排(读状态+跑+汇报)≈ 调度 token。轮间零等待 → 单位时间轮次更多 → token 开销与吞吐正比。
- **停**:用户说「停 loop」→ Claude 不启动下一轮,链断。
- 会话级:关对话链断(进度保留在 checkpoint)。
- 如需固定间隔(低频省 token),仍可用 CronCreate(见本节历史版本)或 OS cron(§A)。

### C. 物理隔离 sub-agent —— 最强 anti-Goodhart(可选强化)

指南 §8 的理想架构:生成/审查/验证拆成**独立 sub-agent**,通过**临时 JSON 单向传递**,生成端进程级看不到任何回测指标。
- `generate_agent`:加载因子库 → 演化/LLM → 写 `cache/pipeline/candidates.json`(纯表达式)。
- `review_agent`:读 candidates → 四过滤+LLM 精判 → 写 `reviewed.json`。
- `evaluate_agent`:读 reviewed → 回测+11 过滤+去重 → 更新 `checkpoint.json`。
- 由 Claude `/loop` 或一个 `orchestrator_agent` 串起来(三次 Agent 调用)。

> 当前 `loop_orchestrate.py` 是**单进程逻辑隔离**(同进程,但生成函数代码层面拿不到指标)——
> 对绝大多数场景已足够。物理隔离是 defense-in-depth,但每轮多两次 sub-agent spawn,**token 更贵**。
> 如需,可基于现有 `engine/`、`backtest/`、`filters.py` 拆出三个入口脚本(本文不展开,按需再实现)。

**取舍建议**:生产用 A(最省);探索/调试用 B;只有强烈防拟合诉求才上 C。

---

## 3. 必备配置(任何启动方式)

### 3.1 从 `factor_loop_engine/` 启动 Claude(让 skill/settings 生效)
```powershell
cd C:\Users\Administrator\Desktop\codewhale\factor_loop_engine
claude
```
- `.claude/skills/factor-mining.md` 自动加载(项目背景/算子/阈值/跑法 → 每轮不重复推导,省 token)。
- `.claude/settings.json` 预授权 `uv run` 命令(循环里少弹权限)。

### 3.2 LLM(DeepSeek)—— `.env`
`factor_loop_engine/.env`(gitignored):
```ini
DEEPSEEK_API_KEY=sk-...   DEEPSEEK_MODEL=deepseek-v4-pro
GLM_API_KEY=...           GLM_MODEL=glm-5.2            # 备用
GENERATION_PROVIDER=deepseek   REVIEW_PROVIDER=deepseek
```
切 provider 改最后两行即可(glm/deepseek/mock)。验证:`uv run --directory factor_loop_engine code\smoke_keys.py`。

### 3.3 回测(alphalab)
- 工具目录 `C:\Users\Administrator\Desktop\因子检测操作步骤`(独立 .venv + rqdatac 缓存已预热到 2018-2025)。
- 配置 `my.yaml`(universe 中证全指、行业+市值中性、单边千一)。
- `backtest/alphalab_adapter.py` 自动调用;单次 ≈ 100~120s。

### 3.4 检查点(断点续跑)
`output/checkpoint.json`(原子写):迭代号 / 已测哈希(去重)/ 入库因子(表达式+IC序列+指标+骨架)/ FSA 计数。
任意中断后下次 `run_round_cli.py` 自动 `Checkpoint.load` 续跑。**清零重来 = 删此文件**。

---

## 4. mock vs 真实

| 模式 | 命令 | LLM | 回测 | 用途 |
|---|---|---|---|---|
| mock(离线) | `code/run_round_cli.py --mock --n 100` | 无(llm 分支=随机) | MockEvaluator 假指标 | 验证流程/调试,零成本 |
| 真实 | `code/run_round_cli.py --n 100` | DeepSeek | alphalab | 真实挖因子 |

真实模式首次会缓存字段宽表到 `cache/panels/`(构建一次,之后秒级加载)。

---

## 5. 跑起来后的典型操作

| 想做的事 | 命令 |
|---|---|
| 跑一轮(mock) | `uv run --directory factor_loop_engine code/run_round_cli.py --mock --n 100` |
| 跑一轮(真实) | `uv run --directory factor_loop_engine code/run_round_cli.py --n 100` |
| 看因子库状态 | `uv run --directory factor_loop_engine code/lib_status.py` |
| 看入库因子明细 | 读 `output/checkpoint.json` 的 `stored_factors` |
| **导出最终因子(parquet)** | `uv run --directory factor_loop_engine code/export_factors.py` |
| 重置(全新开始) | 删 `output/checkpoint.json` |
| 清 OSS/面板缓存重算 | 删 `cache/` |
| 跑全部测试 | `uv run --directory factor_loop_engine pytest` |

### 5.1 导出最终因子(你要的 parquet 值表)

入库因子默认只存「表达式 + 指标」在 `output/checkpoint.json`。要拿到**能直接用的因子值宽表**(就是 `gru_factor_ir.parquet` 那种 `date × 股票`、float 的 parquet,可喂回 alphalab 或做策略),跑:

```powershell
uv run --directory factor_loop_engine code/export_factors.py            # 真实数据(默认)
uv run --directory factor_loop_engine code/export_factors.py --mock     # 合成面板(演示格式)
# 可选:--checkpoint <path> 指定别的检查点、--out <dir> 指定输出目录
```

产出 `output/factors/`:
```
0001_zscore_ma_overnight_20_<hash>.parquet     ← 因子值宽表(date 列 + 每股一列,float32,alphalab 同款)
0002_div_sub_adj_close_..._<hash>.parquet
manifest.csv                                    ← idx/file/expr/direction/ic_mean/icir/ls_annual/ls_sharpe/calmar/long_excess_annual
```

**完整链路**:真实 loop 攒因子 → `checkpoint.json` → `export_factors.py` → `output/factors/*.parquet`(你要的最终因子文件)。
- 入库为 0 时会提示「先跑真实模式攒因子」。
- 默认在 2018-2025 真实面板上求值(首次构建缓存到 `cache/panels/`,之后秒级);`--mock` 用合成面板只演示格式。
- 导出的 parquet 已验证与 alphalab 输入格式一致(可直接 `alphalab check <导出文件> -c my.yaml -d <out>` 复核)。

---

## 6. 故障排查

| 现象 | 排查 |
|---|---|
| GLM 报 1113 余额不足 | 该 key 仅免费 flash 可用;用 DeepSeek 或给 GLM 充值(`.env` 切) |
| alphalab 报错/无产出 | rqdatac 缓存是否预热到 2018-2025、universe 配置、独立 .venv 的 alphalab 可执行 |
| 单轮很慢 | 真实模式 = 候选数 × ~110s(回测是瓶颈);减小 `--n` 或只回测过审查的(已默认如此) |
| 入库长期为 0 | 严过滤(每年都过)下正常(研报 0.41%);多跑几轮,或检查字段/前复权是否对 |
| 导入错误 `No module named ...` | 脚本须放 `code/` 根(扁平导入);用 `uv run --directory factor_loop_engine` |

---

## 7. 已知待办 / 可选强化

1. **M4 参数扰动反馈未接循环**:`perturber.observe(键, 窗口, sharpe)` 是设计允许的唯一「指标→生成」通道(用回测 sharpe 引导窗口扰动);当前 perturber 只在 evolve 内部、未喂回 sharpe 历史。接入:在 `loop_orchestrate` 入库时,对入库因子提取时序窗口 + ls_sharpe 调 `perturber.observe`,并 `checkpoint.capture(perturber=...)`。
2. **失败模式库(#11)自动累积**:当前默认空;可把连续低 IC 的 hash 加入 `failed_hashes`。
3. **回测并行**:alphalab 单次 ~110s,串行回测是瓶颈;可改多进程并行(注意 rqdatac 并发)。
4. **物理隔离 sub-agent(§2-C)**:按需实现。

---

## 8. 验收对照(指南 §5 阶段 7)

| 项 | 状态 |
|---|---|
| skill 怎么写 | ✅ `.claude/skills/factor-mining.md` |
| `/loop` 怎么调度 | ✅ §2-B(连续调度:完成即启动下一轮)+ §2-A(OS cron 最省) |
| hook 怎么配 | ✅ `.claude/settings.json`(预授权)+ `code/lib_status.py`(状态摘要,可挂 Stop hook) |
| 检查点怎么放 | ✅ `output/checkpoint.json`(原子,断点续跑) |
| sub-agent 怎么拆 | ✅ §2-C(物理隔离方案,可选) |
| **照此能组装出定时闭环** | ✅ §2-A 命令可直接落地 |

---

核心算法层 + 对接方案全部就绪。**mock 模式 `run_round_cli.py --mock` 已验证可跑**;真实模式接好 DeepSeek + alphalab 即可生产。

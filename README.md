# factor_loop_engine —— Loop+Engineering 自动化因子发现引擎

基于中金《大模型系列(7):基于 Loop+Engineering 的自动化因子发现引擎》。
**核心算法层纯 Python**(131 单元测试),OSS+duckdb 取数,LLM 可切换 provider,真实回测接 alphalab。

## 文档
- 执行总纲:`docs/项目执行指南.md`(研报 M1–M9 拆解、7 阶段规划、关键参数)
- 取数背景:`docs/OSS取数说明.md`(阿里云 OSS + DuckDB + RAM 角色)
- Claude Code 对接/定时闭环:`output/claude_code_integration.md`

## 快速开始

```powershell
uv sync --directory factor_loop_engine                                    # 装依赖(首次)
uv run --directory factor_loop_engine pytest                              # 131 测试
uv run --directory factor_loop_engine code/run_round_cli.py --mock --n 100 # 跑一轮(离线 mock,不花钱)
uv run --directory factor_loop_engine code/lib_status.py                   # 看因子库状态
uv run --directory factor_loop_engine code/export_factors.py               # 导出入库因子为 parquet
```
真实模式(DeepSeek 生成 + alphalab 回测):配好 `.env` 后去掉 `--mock`。

## 目录结构(四分离:代码 / 缓存 / 产出 / 文档)

```
factor_loop_engine/
├── code/        💻 源代码(engine/ data_layer/ llm/ backtest/ tests/ + 编排/CLI 脚本)
├── config/      ⚙️ alphalab.yaml(项目专用回测配置)
├── docs/        📄 规格(项目执行指南、OSS取数说明)
├── output/      📊 产出(claude_code_integration.md、lessons.md、运行时 checkpoint/factors)
├── cache/       📦 可重建缓存(OSS 行情/面板 parquet;gitignore)
├── .env(.example)   LLM key + alphalab 路径(.env gitignore,.example 是模板)
└── pyproject.toml / uv.lock
```

## 新机器部署(可移植性)

| 依赖 | 说明 |
|---|---|
| **Python + uv** | Python ≥3.11;`uv sync` 装依赖(`pyproject.toml`) |
| **数据(OSS)** | 取数靠阿里云 ECS 实例元数据(IMDS)。**两种办法**:① 新机器也是阿里云 ECS(同 region);② **把 `cache/` 整个拷过去**——加载器优先读本地缓存,有就不碰 OSS,无需改代码 |
| **回测(alphalab)** | 外部工具(独立 venv + rqdatac 凭证 + 缓存预热)。路径用环境变量 `ALPHALAB_DIR` 配(见 `.env`),换机器改这一处 |
| **LLM** | 复制 `.env.example` → `.env`,填 DeepSeek/GLM key;切 provider 改 `GENERATION_PROVIDER`/`REVIEW_PROVIDER` |
| **运行** | `uv sync` → 配 `.env` → `uv run --directory factor_loop_engine code/run_round_cli.py --n 30` |

> `package=false`(uv 虚拟项目):规避中文路径写进 venv `.pth` 致 site 崩溃。导入用扁平包:`from data_layer import oss`(`uv run code/x.py` 自动把 `code/` 放进 `sys.path[0]`)。

## 关键参数
全部集中在 `code/engine/config.py`,每条标【出处】([研报]/[推断]/[默认]/[用户]):
算子 14 / 深度≤4 / 演化预算 25·25·15·15·20 / 窗口集 {5,10,20,40,60,80,120,250} / FSA 15%·2·5 /
回测 2018-2025 horizon=5 / 13 项过滤(含多头超额>0、单调性>0.85)/ warmup 从 2015 起。

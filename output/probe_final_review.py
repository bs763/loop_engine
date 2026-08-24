# -*- coding: utf-8 -*-
"""诊断探针:实测 LLM 终审的原始输出与 parse_verdict 判定是否一致。一次性脚本。"""
import sys

sys.path.insert(0, "code")

from engine.expression import parse
from engine.checkpoint import Checkpoint
from llm.settings import review_provider
from llm.mechanisms import build_review_prompt, parse_verdict

provider = review_provider()
cp = Checkpoint.load("output/checkpoint.json")

# 案例 1-2:真实在库因子(优质,应 ACCEPT)
# 案例 3:单年依赖构造(逐年:一年 +40%,其余 +1% —— mblock 规则①必须拒)
cases = []
for f in cp.stored_factors[:2]:
    m = dict(f["metrics"]); m["annual_ls_return"] = {y: 0.08 for y in range(2018, 2026)}
    cases.append(("库存优质因子", f["expr"], m))
bad = {"ic_mean": 0.05, "icir": 0.6, "ls_sharpe": 2.0, "calmar": 1.5,
       "long_excess_annual": 0.02, "monotonicity": 0.9,
       "annual_ls_return": {2018: 0.01, 2019: 0.012, 2020: 0.008, 2021: 0.40,
                            2022: 0.011, 2023: 0.009, 2024: 0.013, 2025: 0.01}}
cases.append(("单年依赖构造(应拒)", cp.stored_factors[0]["expr"], bad))

for label, expr, m in cases:
    prompt = build_review_prompt(parse(expr), metrics=m)
    try:
        raw = provider.complete(prompt, temperature=0.1)
    except Exception as e:
        print(f"[{label}] API错: {e}")
        continue
    accept, why = parse_verdict(raw)
    print(f"[{label}] parse判={'放行' if accept else '拒绝'}")
    print(f"  原文: {raw[:220]!r}")
    print()

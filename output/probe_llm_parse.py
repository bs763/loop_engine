# -*- coding: utf-8 -*-
"""诊断探针:连续调 LLM 生成端,统计解析失败率并打印失败原文与失败阶段。一次性脚本。"""
import re
import sys

sys.path.insert(0, "code")

import numpy as np

from llm.settings import generation_provider
from llm.mechanisms import MECHANISMS, build_generation_prompt, _first_balanced_sexpr
from engine.expression import parse

FIELDS = ["adj_close", "adj_high", "adj_low", "overnight", "intraday", "amplitude",
          "up_shadow", "down_shadow", "hl_ratio", "ret", "log_volume", "log_amount", "log_mv"]

provider = generation_provider()
rng = np.random.default_rng(0)
N = 15
ok = bad = api_err = 0

for i in range(N):
    mech = MECHANISMS[int(rng.integers(0, len(MECHANISMS)))]
    prompt = build_generation_prompt(mech, FIELDS)
    try:
        text = provider.complete(prompt, temperature=0.8)
    except Exception as e:
        api_err += 1
        print(f"[{i}] API错 {type(e).__name__}: {e}")
        continue
    # 复刻 extract_expression 的各阶段,定位失败原因
    m = re.search(r"```(?:[a-zA-Z]*)?\s*(.*?)\s*```", text, re.S)
    candidate = m.group(1) if m else text
    s = _first_balanced_sexpr(candidate)
    if s is None:
        leaf = re.search(r"\b([a-zA-Z_]\w+)\b", candidate)
        s = leaf.group(1) if leaf else None
    stage = None
    if not s:
        stage = "无表达式"
    else:
        try:
            node = parse(s)
            node.validate()
        except Exception as e:
            stage = f"parse/validate: {type(e).__name__}: {str(e)[:80]}"
        else:
            if not node.fields() or not node.fields().issubset(set(FIELDS)):
                stage = f"字段越界: {sorted(node.fields() - set(FIELDS))}"
            elif node.is_leaf():
                stage = "裸叶子"
    if stage is None:
        ok += 1
    else:
        bad += 1
        print(f"[{i}] 失败({mech['id']}) 阶段={stage}")
        print(f"    原文前400字: {text[:400]!r}")
        print()

print(f"=== 探针结果: ok={ok} 解析失败={bad} API错={api_err} 共{N} 失败率={bad/max(ok+bad,1)*100:.0f}% ===")

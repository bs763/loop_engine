# -*- coding: utf-8 -*-
"""LLM 连通性验证:对生成端/审查端各发 1 条最小请求,验证 key + 模型名(按 .env 当前分配)。

跑法:  uv run code/smoke_keys.py
失败会打印 HTTP 状态与 API 错误体,便于定位(多半是模型名 / endpoint 问题)。
"""
from __future__ import annotations

import requests

from llm.settings import generation_provider, review_provider


def smoke(name: str, provider, prompt: str = "只回复两个字:在的") -> bool:
    try:
        out = provider.complete(prompt, temperature=0.1)
        print(f"[{name}] OK → {out[:80]!r}")
        return True
    except requests.HTTPError as e:
        resp = e.response
        code = resp.status_code if resp is not None else "?"
        body = (resp.text[:400] if resp is not None else "(无响应体)")
        print(f"[{name}] HTTP {code} → {body}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] ERROR → {type(e).__name__}: {e}")
        return False


def main() -> None:
    print("=== LLM 连通性验证(各 1 条最小请求)===")
    ok_gen = smoke("生成端", generation_provider())
    ok_rev = smoke("审查端", review_provider())
    print(f"\n结果:生成端={'OK' if ok_gen else 'FAIL'}   审查端={'OK' if ok_rev else 'FAIL'}")


if __name__ == "__main__":
    main()

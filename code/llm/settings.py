# -*- coding: utf-8 -*-
"""LLM 运行配置:从 .env 加载密钥,构造生成端/审查端 provider。

当前分配(2026-08-17 起,由 .env 的 GENERATION_PROVIDER / REVIEW_PROVIDER 控制):
生成(机制引导)与审查(入库前 LLM 终审,仅 1-2 次/轮)均默认 glm——走智谱
Anthropic 兼容端点(Coding Plan 订阅,与 Claude Code 同 key);deepseek 保留备用。

.env(factor_loop_engine/.env,gitignored):
  GLM_API_KEY / GLM_MODEL          智谱(Anthropic 端点,Coding Plan)
  DEEPSEEK_API_KEY / DEEPSEEK_MODEL 备用

用法:
  from llm.settings import generation_provider, review_provider
  node = generate_expression(generation_provider(), fields)
  accept, reason = review_expression(review_provider(), node)  # LLM 终审(过滤16)
"""
from __future__ import annotations

import os

from paths import PROJECT_ROOT

from llm.provider import get_provider

_ENV_PATH = PROJECT_ROOT / ".env"


def _load_env(path) -> None:
    """简易 .env 加载:KEY=VALUE 写进 os.environ(不覆盖已有)。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# 导入即加载 .env(供 provider.get_provider 读 GLM_API_KEY / DEEPSEEK_API_KEY)
_load_env(_ENV_PATH)


def _build(name: str):
    """按名构造 provider(glm / deepseek / mock)。模型名从对应 env 读。"""
    if name == "glm":
        # 走智谱 Anthropic 兼容端点(Coding Plan 订阅,与 Claude Code 同 key;
        # v4 按量端点余额独立,空余额报 1113)。glm-5.x 带思考、生成端 prompt 长,
        # 超时放宽(默认 120s,GLM_TIMEOUT 可调),max_tokens 默认 4096(GLM_MAX_TOKENS 可调)
        return get_provider("glm-anthropic", model=os.environ.get("GLM_MODEL", "glm-5.3"),
                            timeout=int(os.environ.get("GLM_TIMEOUT", "120")),
                            max_tokens=int(os.environ.get("GLM_MAX_TOKENS", "4096")))
    if name == "deepseek":
        return get_provider("deepseek", model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    if name == "mock":
        return get_provider("mock")
    raise ValueError(f"未知 provider 名: {name}(可选 glm/deepseek/mock)")


def generation_provider():
    """生成端(机制引导,token 大头)。默认 GLM;改 .env 的 GENERATION_PROVIDER 可切。"""
    return _build(os.environ.get("GENERATION_PROVIDER", "glm"))


def review_provider():
    """审查端(精判,token 小头)。默认 DeepSeek;改 .env 的 REVIEW_PROVIDER 可切。"""
    return _build(os.environ.get("REVIEW_PROVIDER", "deepseek"))

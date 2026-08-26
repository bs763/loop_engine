# -*- coding: utf-8 -*-
"""LLM 运行配置:从 .env 加载密钥,构造生成端/审查端 provider。

当前分配(2026-08-25 终审端切换,由 .env 的 GENERATION_PROVIDER / REVIEW_PROVIDER 控制):
生成(机制引导,token 大头)= glm(智谱 Anthropic 兼容端点,Coding Plan 订阅);
审查(入库前 LLM 终审,仅 1-2 次/轮)= deepseek(deepseek-v4-pro,用户 2026-08-25 拍板
——GLM 终审 5 裁 2 误判后升级模型;单次 ~20s,DEEPSEEK_TIMEOUT 默认 120s)。
kimi 分支保留可选(2026-08-19 试切后弃用:该 key 组织级 RPM=3 太慢)。

.env(factor_loop_engine/.env,gitignored):
  GLM_API_KEY / GLM_MODEL          智谱(Anthropic 端点,Coding Plan)
  DEEPSEEK_API_KEY / DEEPSEEK_MODEL 备用
  KIMI_API_KEY / KIMI_MODEL        Kimi 备用(勿切回,RPM=3 太慢)

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


# 导入即加载 .env(供 provider.get_provider 读 GLM_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY)
_load_env(_ENV_PATH)


def _build(name: str):
    """按名构造 provider(glm / deepseek / kimi / mock)。模型名从对应 env 读。"""
    if name == "glm":
        # 走智谱 Anthropic 兼容端点(Coding Plan 订阅,与 Claude Code 同 key;
        # v4 按量端点余额独立,空余额报 1113)。glm-5.x 带思考、生成端 prompt 长,
        # 超时放宽(默认 120s,GLM_TIMEOUT 可调),max_tokens 默认 4096(GLM_MAX_TOKENS 可调)
        return get_provider("glm-anthropic", model=os.environ.get("GLM_MODEL", "glm-5.3"),
                            timeout=int(os.environ.get("GLM_TIMEOUT", "120")),
                            max_tokens=int(os.environ.get("GLM_MAX_TOKENS", "4096")))
    if name == "deepseek":
        # 终审端切 DeepSeek(用户 2026-08-25):v4-pro 带思考、单次实测 ~20s,
        # 默认 30s 超时过紧 → DEEPSEEK_TIMEOUT 可调(默认 120s)
        return get_provider("deepseek", model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
                            timeout=int(os.environ.get("DEEPSEEK_TIMEOUT", "120")))
    if name == "kimi":
        # 月之暗面 Moonshot,OpenAI 兼容端点(https://api.moonshot.cn/v1)。
        # 注意:2026-08-19 试切弃用——该 key 组织级 RPM=3(每分钟 3 次)太慢;
        # kimi-k2 系列只接受 temperature=1(provider 已强制固定)。
        return get_provider("kimi", model=os.environ.get("KIMI_MODEL", "kimi-k2.6"),
                            timeout=int(os.environ.get("KIMI_TIMEOUT", "120")),
                            max_tokens=int(os.environ.get("KIMI_MAX_TOKENS", "4096")))
    if name == "mock":
        return get_provider("mock")
    raise ValueError(f"未知 provider 名: {name}(可选 glm/deepseek/kimi/mock)")


def generation_provider():
    """生成端(机制引导,token 大头)。默认 GLM;改 .env 的 GENERATION_PROVIDER 可切。"""
    return _build(os.environ.get("GENERATION_PROVIDER", "glm"))


def review_provider():
    """审查端(精判,token 小头)。默认 GLM;改 .env 的 REVIEW_PROVIDER 可切。"""
    return _build(os.environ.get("REVIEW_PROVIDER", "glm"))

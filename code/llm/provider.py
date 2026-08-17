# -*- coding: utf-8 -*-
"""LLM provider:统一接口 + OpenAI 兼容实现(DeepSeek 生成 / 智谱 GLM 审查)+ Mock。

指南 §4 决策:生成端默认 DeepSeek-v4,机制引导 + 审查精判用智谱 GLM;调用统一封装为可切换 provider。
本模块把「调谁、怎么调」解耦:所有 provider 实现 complete(prompt) -> str,上层(mechanisms.py)不关心具体厂商。

API key 通过环境变量注入(阶段 4 用 Mock 跑通,真实 key 由用户提供):
  DeepSeek: DEEPSEEK_API_KEY   (base_url https://api.deepseek.com)
  智谱 GLM: GLM_API_KEY        (base_url https://open.bigmodel.cn/api/paas/v4,OpenAI 兼容)
"""
from __future__ import annotations

import os

import requests


class LLMProvider:
    """provider 抽象基类。子类实现 complete。"""

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> str:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """离线/测试用:返回固定响应(或按 callable 动态生成)。不联网。"""

    def __init__(self, response: str = "zscore(ma(close, 20))",
                 responder=None):
        self.response = response
        self.responder = responder  # callable(prompt)->str,优先于 response

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> str:
        if self.responder is not None:
            return self.responder(prompt)
        return self.response


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions 兼容 provider(DeepSeek / 智谱 GLM 等均支持)。"""

    def __init__(self, base_url: str, api_key: str | None, model: str,
                 default_temperature: float = 0.7, timeout: int = 30,
                 system_default: str = "你是资深 A 股量化研究员,只按要求输出。"):
        if not api_key:
            raise ValueError(
                f"缺少 api_key(base_url={base_url})。请设置对应环境变量或显式传入。")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.timeout = timeout
        self.system_default = system_default

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> str:
        messages = [
            {"role": "system", "content": system or self.system_default},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicMessagesProvider(LLMProvider):
    """Anthropic Messages 兼容 provider(智谱 GLM Coding Plan 走 /api/anthropic,与 Claude Code 同端点同 key)。

    与 OpenAICompatibleProvider 的差异:POST {base}/v1/messages;system 是顶层字段;
    max_tokens 必填;响应正文在 content 数组的 text 块里(思考块 type=thinking,不含正文,跳过)。
    注意:该端点走 Coding Plan 订阅额度,/api/paas/v4 按量余额是独立计费,互不相通。
    """

    def __init__(self, base_url: str, api_key: str | None, model: str,
                 default_temperature: float = 0.7, timeout: int = 120,
                 max_tokens: int = 4096,
                 system_default: str = "你是资深 A 股量化研究员,只按要求输出。"):
        if not api_key:
            raise ValueError(
                f"缺少 api_key(base_url={base_url})。请设置对应环境变量或显式传入。")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.system_default = system_default

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system or self.system_default,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.default_temperature if temperature is None else temperature,
            "stream": False,
        }
        url = f"{self.base_url}/v1/messages"
        resp = requests.post(
            url,
            headers={"x-api-key": self.api_key,
                     "Authorization": f"Bearer {self.api_key}",
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


# ---- 厂商预设 ----
_PRESETS = {
    # name: (base_url, env_key, default_model)
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY", "glm-4-plus"),
    # 智谱 Anthropic 兼容端点(Coding Plan 订阅额度;v4 按量端点余额独立,报 1113 就切这个)
    "glm-anthropic": ("https://open.bigmodel.cn/api/anthropic", "GLM_API_KEY", "glm-5.3"),
}


def get_provider(name: str, **kw) -> LLMProvider:
    """按名取 provider。mock / deepseek / glm / glm-anthropic / openai(通用,需传 base_url+api_key+model)。"""
    if name == "mock":
        return MockProvider(response=kw.get("response", "zscore(ma(close, 20))"),
                            responder=kw.get("responder"))
    if name in _PRESETS:
        base_url, env_key, default_model = _PRESETS[name]
        api_key = kw.pop("api_key", None) or os.environ.get(env_key)
        model = kw.pop("model", default_model)
        cls = AnthropicMessagesProvider if name == "glm-anthropic" else OpenAICompatibleProvider
        return cls(base_url=base_url, api_key=api_key, model=model, **kw)
    if name == "openai":
        return OpenAICompatibleProvider(**kw)  # 调用方负责传 base_url/api_key/model
    raise ValueError(f"未知 provider: {name}(可选 mock/deepseek/glm/openai)")

# -*- coding: utf-8 -*-
"""LLM provider:统一接口 + OpenAI/Anthropic 兼容实现 + Mock。

当前配置(2026-08-19 恢复,见 .env):生成端与审查端均走智谱 GLM-5.x(Coding Plan,
Anthropic 兼容端点 glm-anthropic);DeepSeek 配置保留备用(OpenAI 兼容,按量计费)。
Kimi 预设保留可选(2026-08-19 试切后弃用:该 key 组织级 RPM=3 太慢)。
本模块把「调谁、怎么调」解耦:所有 provider 实现 complete(prompt) -> str,上层(mechanisms.py)不关心具体厂商。

API key 通过环境变量注入:
  智谱 GLM: GLM_API_KEY   (Anthropic 端点 /api/anthropic = Coding Plan;v4 端点 = 按量,计费独立)
  DeepSeek: DEEPSEEK_API_KEY (base_url https://api.deepseek.com)
  Kimi:     KIMI_API_KEY   (base_url https://api.moonshot.cn/v1,OpenAI 兼容;
                            该 key org 级 RPM=3,kimi preset 内置 min_interval=25s + 429 退避;
                            kimi-k2 系列只接受 temperature=1)
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
    """OpenAI Chat Completions 兼容 provider(DeepSeek / 智谱 GLM / Kimi 等均支持)。

    fixed_temperature: 部分模型(kimi-k2 系列)只接受 temperature=1,任何其它值(含调用方
    显式传入)都会被 400 拒绝 → 设置后 complete 强制用该值,忽略调用方参数。
    min_interval: 两次请求间最小间隔秒(限流保护;如 kimi org RPM=3 → 设 25,确保 ≤2/min)。
    retry_on_429: 命中 429 限流时按错误体提示的秒数等待后重试的最大次数(0=不重试)。
    """

    def __init__(self, base_url: str, api_key: str | None, model: str,
                 default_temperature: float = 0.7, timeout: int = 30,
                 max_tokens: int | None = None,
                 fixed_temperature: float | None = None,
                 min_interval: float = 0.0, retry_on_429: int = 0,
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
        self.fixed_temperature = fixed_temperature
        self.min_interval = min_interval
        self.retry_on_429 = retry_on_429
        self.system_default = system_default
        self._last_call_ts = 0.0
        self._429_waits = 0      # 累计限流等待(秒),供诊断

    def _throttle(self) -> None:
        """保证两次 complete 间隔 ≥ min_interval(进程内)。"""
        import time
        if self.min_interval > 0:
            wait = self.min_interval - (time.time() - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)
        self._last_call_ts = time.time()

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None) -> str:
        import re
        import time
        messages = [
            {"role": "system", "content": system or self.system_default},
            {"role": "user", "content": prompt},
        ]
        # fixed_temperature 非 None → 忽略调用方 temperature(kimi-k2 系列只许 1)
        if self.fixed_temperature is not None:
            temp = self.fixed_temperature
        else:
            temp = self.default_temperature if temperature is None else temperature
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        url = f"{self.base_url}/chat/completions"
        self._throttle()
        for attempt in range(self.retry_on_429 + 1):
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=self.timeout,
            )
            if resp.status_code == 429 and attempt < self.retry_on_429:
                wait = 1.0
                m = re.search(r"after (\d+) seconds?", resp.text)
                if m:
                    wait = float(m.group(1))
                self._429_waits += wait
                time.sleep(wait)
                continue
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
    # Kimi(月之暗面 Moonshot,OpenAI 兼容;2026-08-19 起为默认生成+审查端)
    "kimi": ("https://api.moonshot.cn/v1", "KIMI_API_KEY", "kimi-latest"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY", "glm-4-plus"),
    # 智谱 Anthropic 兼容端点(Coding Plan 订阅额度;v4 按量端点余额独立,报 1113 就切这个)
    "glm-anthropic": ("https://open.bigmodel.cn/api/anthropic", "GLM_API_KEY", "glm-5.3"),
}


def get_provider(name: str, **kw) -> LLMProvider:
    """按名取 provider。mock / kimi / deepseek / glm / glm-anthropic / openai(通用,需传 base_url+api_key+model)。"""
    if name == "mock":
        return MockProvider(response=kw.get("response", "zscore(ma(close, 20))"),
                            responder=kw.get("responder"))
    if name in _PRESETS:
        base_url, env_key, default_model = _PRESETS[name]
        api_key = kw.pop("api_key", None) or os.environ.get(env_key)
        model = kw.pop("model", default_model)
        if name == "glm-anthropic":
            return AnthropicMessagesProvider(base_url=base_url, api_key=api_key, model=model, **kw)
        if name == "kimi":
            # kimi-k2 系列只接受 temperature=1,强制固定,忽略调用方显式温度(否则 400)
            kw.setdefault("fixed_temperature", 1.0)
            # 该 key 组织级 RPM=3(实测 2026-08-19)→ 请求间隔 ≥25s(≤2/min 留余量)+ 429 退避重试
            kw.setdefault("min_interval", 25.0)
            kw.setdefault("retry_on_429", 3)
        return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model, **kw)
    if name == "openai":
        return OpenAICompatibleProvider(**kw)  # 调用方负责传 base_url/api_key/model
    raise ValueError(f"未知 provider: {name}(可选 mock/kimi/deepseek/glm/openai)")

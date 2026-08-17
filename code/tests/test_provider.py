# -*- coding: utf-8 -*-
"""provider.py 单元测试:Mock、工厂切换、key 缺失、OpenAI 兼容请求构造(不发真实请求)。"""
import pytest

from llm import provider as P


# ---------------- MockProvider ----------------

def test_mock_returns_fixed_response():
    m = P.MockProvider(response="div(ma(close, 20), std(high, 10))")
    assert m.complete("任意 prompt") == "div(ma(close, 20), std(high, 10))"


def test_mock_responder_overrides_response():
    m = P.MockProvider(responder=lambda prompt: "REJECT: 测试")
    assert m.complete("...").startswith("REJECT")


# ---------------- 工厂切换(能切 provider) ----------------

def test_factory_returns_mock():
    assert isinstance(P.get_provider("mock"), P.MockProvider)


def test_factory_returns_deepseek_with_preset():
    prov = P.get_provider("deepseek", api_key="sk-test")
    assert isinstance(prov, P.OpenAICompatibleProvider)
    assert prov.base_url == "https://api.deepseek.com"
    assert prov.api_key == "sk-test"
    assert prov.model == "deepseek-chat"


def test_factory_returns_glm_with_preset():
    prov = P.get_provider("glm", api_key="glm-test")
    assert isinstance(prov, P.OpenAICompatibleProvider)
    assert prov.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert prov.model == "glm-4-plus"


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        P.get_provider("nope")


# ---------------- key 缺失 ----------------

def test_deepseek_without_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        P.get_provider("deepseek")


def test_deepseek_reads_env_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    prov = P.get_provider("deepseek")
    assert prov.api_key == "from-env"


# ---------------- OpenAI 兼容请求构造(monkeypatch,不联网) ----------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


def test_openai_compat_request_shape(monkeypatch):
    captured = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResp({"choices": [{"message": {"content": "zscore(close)"}}]})
    monkeypatch.setattr(P.requests, "post", fake_post)

    prov = P.get_provider("deepseek", api_key="sk-x")
    out = prov.complete("生成一个因子", temperature=0.9)

    assert out == "zscore(close)"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-x"
    assert captured["json"]["model"] == "deepseek-chat"
    assert captured["json"]["temperature"] == 0.9
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][1]["content"] == "生成一个因子"

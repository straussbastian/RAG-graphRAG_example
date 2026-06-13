import graph_viz.client as client_mod
from graph_viz.client import LightRAGClient


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_get_graph_uses_x_api_key_only(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.update(url=url, params=params, headers=headers)
        return _Resp({"nodes": [], "edges": []})

    monkeypatch.setattr(client_mod.httpx, "get", fake_get)
    c = LightRAGClient("https://x.y/", "secret")
    out = c.get_graph()
    assert out == {"nodes": [], "edges": []}
    assert captured["url"] == "https://x.y/graphs"
    assert captured["headers"] == {"X-API-Key": "secret"}
    assert "Authorization" not in captured["headers"]
    assert captured["params"]["label"] == "*"


def test_query_data_posts_only_need_context(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _Resp({"data": {}, "metadata": {}})

    monkeypatch.setattr(client_mod.httpx, "post", fake_post)
    c = LightRAGClient("https://x.y", "secret")
    c.query_data("hallo", mode="local", top_k=5)
    assert captured["url"] == "https://x.y/query/data"
    assert captured["json"] == {
        "query": "hallo", "mode": "local", "only_need_context": True, "top_k": 5,
    }
    assert captured["headers"] == {"X-API-Key": "secret"}

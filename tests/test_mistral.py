from qdrant_viz import mistral


class _FakeEmbeddings:
    def create(self, model, inputs):
        assert model == "mistral-embed"
        assert isinstance(inputs, list) and inputs == ["hello"]

        class _Item:
            embedding = [0.1] * 1024

        class _Resp:
            data = [_Item()]

        return _Resp()


class _FakeClient:
    def __init__(self, api_key):
        assert api_key == "k"
        self.embeddings = _FakeEmbeddings()


def test_embed_text_returns_1024_vector(monkeypatch):
    monkeypatch.setattr(mistral, "Mistral", _FakeClient)
    vec = mistral.embed_text("hello", api_key="k")
    assert len(vec) == 1024
    assert vec[0] == 0.1

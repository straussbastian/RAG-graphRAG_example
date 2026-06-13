import qdrant_viz.mistral as mistral_mod
from qdrant_viz.mistral import embed_texts


class _Item:
    def __init__(self, emb):
        self.embedding = emb


def test_embed_texts_single_call_in_order(monkeypatch):
    captured = {}

    class _Emb:
        def create(self, model, inputs):
            captured["model"] = model
            captured["inputs"] = inputs
            return type("R", (), {"data": [_Item([float(i)] * 3) for i in range(len(inputs))]})()

    class _Client:
        def __init__(self, api_key):
            assert api_key == "k"
            self.embeddings = _Emb()

    monkeypatch.setattr(mistral_mod, "Mistral", _Client)
    out = embed_texts(["x", "y"], api_key="k", model="mistral-embed")

    assert captured["model"] == "mistral-embed"
    assert captured["inputs"] == ["x", "y"]            # one call, both inputs
    assert out == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]   # one vector per input, in order

"""Mistral embedding wrapper. Must use the SAME model that created the stored
vectors (mistral-embed, 1024-dim) or neighbors are meaningless."""
from __future__ import annotations

try:  # mistralai v1 layout
    from mistralai import Mistral
except ImportError:  # mistralai v2.x layout exposes the client under a subpackage
    from mistralai.client import Mistral


def embed_text(text: str, api_key: str, model: str = "mistral-embed") -> list[float]:
    client = Mistral(api_key=api_key)
    resp = client.embeddings.create(model=model, inputs=[text])
    return list(resp.data[0].embedding)


def embed_texts(texts: list[str], api_key: str, model: str = "mistral-embed") -> list[list[float]]:
    """Embed several texts in a single call; returns one vector per input, in order."""
    client = Mistral(api_key=api_key)
    resp = client.embeddings.create(model=model, inputs=texts)
    return [list(item.embedding) for item in resp.data]


def chat_complete(prompt: str, api_key: str, model: str = "mistral-small-latest") -> str:
    """Single-turn chat completion for the pipeline demo's generation step."""
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content

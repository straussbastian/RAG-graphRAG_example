from graph_viz.transform import build_graph_json, build_highlight


def test_build_graph_json_shapes_and_degree():
    raw = {
        "nodes": [
            {"id": "A", "properties": {"entity_type": "concept", "description": "da"}},
            {"id": "B", "properties": {"entity_type": "artifact", "description": "db"}},
            {"id": "C", "properties": {}},
        ],
        "edges": [
            {"source": "A", "target": "B", "properties": {"description": "ab"}},
            {"source": "A", "target": "C", "properties": {}},
        ],
    }
    doc = build_graph_json(raw)
    assert {n["id"] for n in doc["nodes"]} == {"A", "B", "C"}
    deg = {n["id"]: n["degree"] for n in doc["nodes"]}
    assert deg == {"A": 2, "B": 1, "C": 1}
    a = next(n for n in doc["nodes"] if n["id"] == "A")
    assert a["type"] == "concept" and a["description"] == "da"
    c = next(n for n in doc["nodes"] if n["id"] == "C")
    assert c["type"] == "UNKNOWN"
    assert doc["links"][0] == {"source": "A", "target": "B", "description": "ab"}
    assert "concept" in doc["types"] and "artifact" in doc["types"]


def test_build_highlight_maps_and_skips_missing():
    qr = {
        "data": {
            "entities": [
                {"entity_name": "Betriebsmittel"},
                {"entity_name": "Lizard"},
                {"entity_name": "Ghost"},
            ],
            "relationships": [
                {"src_id": "Betriebsmittel", "tgt_id": "Lizard"},
                {"src_id": "Betriebsmittel", "tgt_id": "Ghost"},
            ],
            "chunks": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
        },
        "metadata": {
            "query_mode": "mix",
            "keywords": {"high_level": ["Pruefauftrag"], "low_level": ["Betriebsmittel"]},
        },
    }
    node_ids = {"Betriebsmittel", "Lizard"}
    h = build_highlight(qr, node_ids)
    assert set(h["highlight_nodes"]) == {"Betriebsmittel", "Lizard"}
    assert h["highlight_edges"] == [["Betriebsmittel", "Lizard"]]
    assert h["missing"] == ["Ghost"]
    assert h["keywords"]["high_level"] == ["Pruefauftrag"]
    assert h["mode"] == "mix"
    assert h["chunks"] == 3


def test_build_highlight_handles_empty():
    h = build_highlight({}, set())
    assert h["highlight_nodes"] == [] and h["highlight_edges"] == []
    assert h["keywords"] == {"high_level": [], "low_level": []}
    assert h["chunks"] == 0

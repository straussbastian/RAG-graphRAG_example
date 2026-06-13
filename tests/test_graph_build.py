import build_graph


def test_render_offline_html_embeds_data_and_lib(tmp_path, monkeypatch):
    fake_lib = tmp_path / "lib.js"
    fake_lib.write_text("/*forcegraph-bundle*/", encoding="utf-8")
    monkeypatch.setattr(build_graph, "VENDOR_JS", fake_lib)

    doc = {
        "nodes": [{"id": "Lizard", "type": "concept", "description": "d", "degree": 3}],
        "links": [],
        "types": ["concept"],
    }
    html = build_graph.render_offline_html(doc)
    assert "<html" in html and "ForceGraph3D" in html
    assert "Lizard" in html                 # data embedded
    assert "/*forcegraph-bundle*/" in html  # vendored lib inlined
    assert "/*LIB*/" not in html            # placeholder replaced
    assert "/*DATA*/" not in html

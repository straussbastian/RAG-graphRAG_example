"""Pure transforms: LightRAG API payloads -> frontend-ready structures.
No I/O, so trivially unit-testable."""
from __future__ import annotations


def build_graph_json(raw: dict) -> dict:
    """Convert a LightRAG /graphs payload into a 3d-force-graph document.

    Returns {'nodes': [{id, type, description, degree}],
             'links': [{source, target, description}],
             'types': [sorted distinct entity_type]}.
    """
    nodes_in = raw.get("nodes") or []
    edges_in = raw.get("edges") or []

    degree: dict = {}
    links = []
    for e in edges_in:
        s, t = e.get("source"), e.get("target")
        if s is None or t is None:
            continue
        degree[s] = degree.get(s, 0) + 1
        degree[t] = degree.get(t, 0) + 1
        props = e.get("properties") or {}
        links.append({"source": s, "target": t, "description": props.get("description", "")})

    nodes = []
    types = set()
    for n in nodes_in:
        nid = n.get("id")
        props = n.get("properties") or {}
        ntype = props.get("entity_type") or "UNKNOWN"
        types.add(ntype)
        nodes.append({
            "id": nid,
            "type": ntype,
            "description": props.get("description", ""),
            "degree": degree.get(nid, 0),
        })
    return {"nodes": nodes, "links": links, "types": sorted(types)}


def build_highlight(query_response: dict, node_ids: set) -> dict:
    """From a /query/data response, return the node ids / edges to light up,
    plus keywords and mode. Entities/edges whose ids are not in node_ids are
    skipped; skipped entity names are reported under 'missing'."""
    data = query_response.get("data") or {}
    meta = query_response.get("metadata") or {}
    entities = data.get("entities") or []
    rels = data.get("relationships") or []
    chunks = data.get("chunks") or []

    highlight_nodes = []
    missing = []
    for ent in entities:
        name = (ent.get("entity_name") or "").strip()
        if name in node_ids:
            highlight_nodes.append(name)
        elif name:
            missing.append(name)

    highlight_edges = []
    for rel in rels:
        s = (rel.get("src_id") or "").strip()
        t = (rel.get("tgt_id") or "").strip()
        if s in node_ids and t in node_ids:
            highlight_edges.append([s, t])

    return {
        "highlight_nodes": highlight_nodes,
        "highlight_edges": highlight_edges,
        "keywords": meta.get("keywords") or {"high_level": [], "low_level": []},
        "mode": meta.get("query_mode", ""),
        "chunks": len(chunks),
        "missing": missing,
    }

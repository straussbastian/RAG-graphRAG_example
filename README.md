# Qdrant Embedding 3D-Visualisierung

3D-Visualisierung der `LizardDocu`-Embeddings (MistralAI `mistral-embed`, 1024-dim)
für eine Präsentation: Eine drehbare Punktwolke zeigt, dass semantisch ähnliche
Chunks im Raum nah beieinander liegen — obwohl alle aus *einem* Dokument stammen.

Zwei Deliverables:

1. **`embeddings.html`** — eigenständige Datei, läuft offline in jedem Browser.
   Drehen/Zoomen/Hover (Chunk-Text + Zeilen), Umschalter UMAP ↔ PCA.
2. **Live-App** (`app.py`) — Suchtext eintippen → wird via `mistral-embed`
   embedded, in Qdrant gesucht, der Query-Punkt landet sichtbar bei seinen
   echten Nachbarn (Verbindungslinien, restliche Wolke dimmt ab).

## 🚀 Schnellstart: Lokale Demo mit Docker

Die komplette Demo (klassisches RAG **und** GraphRAG) läuft lokal in Docker — du
brauchst **nur einen Mistral-API-Key**. Qdrant (Vektor-DB) und LightRAG
(Wissensgraph) laufen als mitgelieferte Container mit vorgebauten Daten; nichts
Externes wird kontaktiert. Mistral wird ausschließlich für Live-Embeddings und
-Antworten genutzt (dein Key).

```bash
cp env.example .env        # dann .env öffnen und MISTRAL_API_KEY eintragen
docker compose up          # startet qdrant + lightrag + die App
```

Dann **http://localhost:8888** öffnen. Optional sichtbar:
- Qdrant-Dashboard: http://localhost:6333/dashboard
- LightRAG-WebUI: http://localhost:9621

Beim ersten Start seedet die App den lokalen Qdrant aus `data/qdrant_seed.json`;
der LightRAG-Wissensgraph ist in `lightrag_storage/` bereits vorgebaut (kein
Ingest-Aufwand, keine Wartezeit).

### `LOKAL`-Schalter
- `LOKAL=true` (Default in `env.example`) → die mitgelieferten lokalen Container.
- `LOKAL=false` → die App spricht mit externen Qdrant-/LightRAG-Diensten; dann
  zusätzlich `QDRANT_URL/_API_KEY` und `LIGHTRAG_URL/_API_KEY` in `.env` setzen
  und die App ohne Compose starten (`uv run uvicorn app:app …`).

---

## Setup (ohne Docker / externe Dienste)

1. `.env` im Projektordner mit:
   ```
   LOKAL=false
   QDRANT_URL=https://dein-qdrant-host
   QDRANT_API_KEY=...
   LIGHTRAG_URL=https://dein-lightrag-host
   LIGHTRAG_API_KEY=...
   MISTRAL_API_KEY=...
   ```
2. Abhängigkeiten installieren: `uv sync`

## Build (einmalig)

```
uv run python build_data.py
```

Zieht alle 249 Vektoren aus Qdrant, reduziert 1024D→3D (UMAP + PCA), clustert
für die Farben (K-Means, k via Silhouette) und schreibt:
- `data/points.json` — die Punktwolke (Koordinaten, Cluster, Text, Zeilen)
- `data/reducer.pkl` — das gespeicherte UMAP-Modell
- `embeddings.html` — die eigenständige statische Visualisierung

## Live-Demo

```
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

Dann http://127.0.0.1:8000 öffnen. Suchbegriff eingeben (z. B. „Listenansicht").

## Tests

```
uv run pytest
```

## Architektur

```
qdrant_viz/
  config.py     # .env laden, QDRANT_URL auf :443 normalisieren, Settings
  mistral.py    # embed_text() via mistral-embed
  data.py       # fetch_points(), reduce_3d() (UMAP+PCA), cluster_labels(), build_points_json()
  placement.py  # weighted_centroid() — platziert den Live-Query-Punkt
build_data.py   # Build-Skript → points.json + reducer.pkl + embeddings.html
app.py          # FastAPI: GET / , /api/points , /api/query
web/            # Plotly.js-Frontend (Plotly lokal gebündelt, kein CDN)
```

## Hinweise

- **Port:** `QDRANT_URL` wird intern auf `:443` normalisiert. Der öffentliche
  Ingress serviert auf 443; `qdrant-client` würde sonst auf 6333 defaulten und
  timeouten. `:6333` in der URL wird automatisch entfernt.
- **Modell:** Die Live-Query nutzt `mistral-embed` — dasselbe Modell, das die
  gespeicherten Vektoren erzeugt hat. Andernfalls wären die Nachbarn bedeutungslos.
- **Offline:** Sowohl `embeddings.html` (Plotly inline) als auch die Live-App
  (Plotly unter `web/plotly.min.js` gebündelt) brauchen im Vortrag kein Internet
  fürs Rendering — nur die Live-Query selbst ruft Mistral/Qdrant.
- **Platzierung:** Der Live-Query-Punkt wird als ähnlichkeitsgewichteter
  Schwerpunkt seiner echten Qdrant-Nachbarn im 3D-Raum platziert (robust, keine
  instabile Einzelpunkt-Projektion). Das `reducer.pkl` ist für eine optionale
  „ehrliche" UMAP-Projektion gespeichert, wird aktuell aber nicht genutzt.

---

# GraphRAG — LightRAG Wissensgraph

3D-Visualisierung des LightRAG-Knowledge-Graphen (~1240 Entities, ~1600 Relationen;
im Frontend auf 1000 Knoten begrenzt) als Gegenstück zur Vektor-Punktwolke:
Farbe = `entity_type`, Knotengröße = Grad (Hubs wie *Lizard*, *Betriebsmittel*,
*Prüfauftrag* treten hervor). Eine Live-Query zeigt, welchen *Teilgraphen* LightRAG
für eine Frage traversiert.

Im **lokalen Docker-Modus** (`LOKAL=true`) ist dieser Graph in `lightrag_storage/`
bereits vorgebaut — der `lightrag`-Container nutzt Mistral als LLM + Embedding.
Im externen Modus (`LOKAL=false`) zusätzlich `LIGHTRAG_URL`, `LIGHTRAG_API_KEY`
in `.env` setzen.

## Build (einmalig)

```
uv run python build_graph.py
```

Zieht den Graphen via LightRAG `GET /graphs`, transformiert ihn und schreibt:
- `data/graph.json` — Nodes/Links/Typen fürs Frontend
- `graph.html` — eigenständige Offline-Visualisierung (3d-force-graph inline)

## Live-View (vereinte Demo)

```
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

Dann http://127.0.0.1:8000 öffnen. Auf `/` liegt der **RAG ⇄ GraphRAG-Umschalter**
(Shell mit iframe); der GraphRAG-View direkt unter `/graph`, die Qdrant-Punktwolke
unter `/qdrant`.

Frage eingeben → via LightRAG `POST /query/data` wird der traversierte Subgraph
ermittelt und hervorgehoben (Entities + Relationen leuchten auf, Rest dimmt ab);
die extrahierten Keywords (high/low-level) erscheinen. Der Modus
(local/global/hybrid/mix) ist umschaltbar.

## Architektur (GraphRAG-Teil)

```
graph_viz/
  config.py     # .env laden, LIGHTRAG_* validieren, GraphSettings
  client.py     # LightRAGClient: GET /graphs, POST /query/data (nur X-API-Key)
  transform.py  # build_graph_json(), build_highlight() — pure, keine I/O
build_graph.py  # Build-Skript → data/graph.json + graph.html
graph_web/      # 3d-force-graph-Frontend (Library lokal gebündelt, kein CDN)
app.py          # vereinte FastAPI: Shell / , /qdrant , /graph , /api/graph[/query]
```

## Hinweise (GraphRAG)

- Auth gegenüber LightRAG ist **ausschließlich** der `X-API-Key`-Header — ein
  zusätzlicher `Authorization: Bearer`-Header führt zu „Invalid token".
- Der Browser spricht nie direkt mit LightRAG: die App proxyt server-seitig,
  der API-Key bleibt im Backend.
- `graph.html` ist vollständig offline (Library + Daten inline, kein CDN).
- Der Query-Subgraph mappt direkt: `entity_name` → Node-ID, `src_id`/`tgt_id` →
  Kante. Entities, die (z. B. durch das Node-Limit) nicht im Graphen sind, werden
  übersprungen statt zu crashen.

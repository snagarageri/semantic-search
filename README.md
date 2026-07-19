# semantic-search

A small Rust service that demonstrates semantic search with an in-memory HNSW-style graph and an Axum HTTP API.

## What it does

- Accepts vectors with an identifier via `POST /index`
- Searches for the nearest neighbors via `POST /search`
- Uses a shared `RwLock` to allow concurrent reads while keeping writes exclusive

## Run it

```bash
cargo run
```

Then try:

```bash
curl http://localhost:3000/health
curl -X POST http://localhost:3000/index \
  -H 'Content-Type: application/json' \
  -d '{"id":"doc_1","vector":[0.1,0.8,0.2]}'

curl -X POST http://localhost:3000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":[0.12,0.79,0.18],"top_k":3}'
```

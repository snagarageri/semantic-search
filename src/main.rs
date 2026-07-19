use axum::{
    extract::State,
    http::StatusCode,
    Json, Router,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, sync::{Arc, RwLock}};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{fmt, EnvFilter};

#[derive(Debug, Clone)]
struct AppState {
    index: Arc<RwLock<HnswIndex>>,
}

#[derive(Debug, Clone, Deserialize)]
struct IndexRequest {
    id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct IndexResponse {
    id: String,
    inserted: bool,
    total_vectors: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SearchResult {
    id: String,
    score: f32,
}

#[derive(Debug, Clone, Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Debug, Clone)]
struct HnswIndex {
    vectors: Vec<(String, Vec<f32>)>,
    graph: Vec<Vec<usize>>,
    max_neighbors: usize,
    max_candidates: usize,
}

impl HnswIndex {
    fn new() -> Self {
        Self {
            vectors: vec![],
            graph: vec![],
            max_neighbors: 4,
            max_candidates: 8,
        }
    }

    fn insert(&mut self, id: String, vector: Vec<f32>) -> usize {
        if vector.is_empty() {
            return 0;
        }

        let index = self.vectors.len();
        self.vectors.push((id.clone(), vector.clone()));
        self.graph.push(Vec::new());

        if index == 0 {
            return 1;
        }

        let mut candidates: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(candidate_index, _)| *candidate_index != index)
            .map(|(candidate_index, (_, existing_vector))| {
                (cosine_similarity(&vector, existing_vector), candidate_index)
            })
            .collect();

        candidates.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });

        let selected_neighbors: Vec<usize> = candidates
            .into_iter()
            .take(self.max_neighbors)
            .map(|(_, candidate_index)| candidate_index)
            .collect();

        for &neighbor in &selected_neighbors {
            self.graph[index].push(neighbor);
            self.graph[neighbor].push(index);
        }

        for neighbor in &selected_neighbors {
            self.prune_node(*neighbor);
        }
        self.prune_node(index);

        1
    }

    fn prune_node(&mut self, node: usize) {
        if self.graph[node].len() <= self.max_neighbors {
            return;
        }

        let mut candidates: Vec<(f32, usize)> = self.graph[node]
            .iter()
            .copied()
            .map(|neighbor| {
                let (_, vector) = &self.vectors[neighbor];
                let (_, current_vector) = &self.vectors[node];
                (cosine_similarity(current_vector, vector), neighbor)
            })
            .collect();

        candidates.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });

        let kept: Vec<usize> = candidates
            .into_iter()
            .take(self.max_neighbors)
            .map(|(_, neighbor)| neighbor)
            .collect();

        self.graph[node] = kept;
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() || query.is_empty() {
            return vec![];
        }

        let mut visited = vec![false; self.vectors.len()];
        let mut current = 0usize;
        let mut candidate_pool = vec![0usize];
        visited[0] = true;

        for _ in 0..self.max_candidates.min(self.vectors.len()) {
            let mut best_neighbor = None;
            let mut best_score = f32::NEG_INFINITY;

            for &neighbor in &self.graph[current] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                let (_, vector) = &self.vectors[neighbor];
                let score = cosine_similarity(query, vector);
                if score > best_score {
                    best_score = score;
                    best_neighbor = Some(neighbor);
                }
                candidate_pool.push(neighbor);
            }

            if let Some(next) = best_neighbor {
                current = next;
            } else {
                break;
            }
        }

        let mut scored: Vec<(f32, String)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(index, _)| candidate_pool.contains(index) || *index == current)
            .map(|(_, (id, vector))| (cosine_similarity(query, vector), id.clone()))
            .collect();

        if scored.len() < top_k {
            let mut remaining: Vec<(f32, String)> = self
                .vectors
                .iter()
                .map(|(id, vector)| (cosine_similarity(query, vector), id.clone()))
                .collect();
            scored.append(&mut remaining);
        }

        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });

        scored
            .into_iter()
            .take(top_k)
            .map(|(score, id)| SearchResult { id, score })
            .collect()
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    let mut dot = 0.0;
    let mut left_norm_sq = 0.0;
    let mut right_norm_sq = 0.0;

    for index in 0..len {
        let left_value = left[index];
        let right_value = right[index];
        dot += left_value * right_value;
        left_norm_sq += left_value * left_value;
        right_norm_sq += right_value * right_value;
    }

    if left_norm_sq == 0.0 || right_norm_sq == 0.0 {
        return 0.0;
    }

    dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt())
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn index_vector(
    State(state): State<AppState>,
    Json(request): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, StatusCode> {
    let mut index = state.index.write().unwrap();
    let inserted = index.insert(request.id.clone(), request.vector);
    Ok(Json(IndexResponse {
        id: request.id,
        inserted: inserted > 0,
        total_vectors: index.vectors.len(),
    }))
}

async fn search_vectors(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let index = state.index.read().unwrap();
    let requested_top_k = request.top_k.unwrap_or(5);
    let top_k = requested_top_k.min(index.vectors.len());
    let results = index.search(&request.query, top_k);
    Ok(Json(SearchResponse { results }))
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/index", post(index_vector))
        .route("/search", post(search_vectors))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[tokio::main]
async fn main() {
    fmt()
        .with_env_filter(EnvFilter::new("semantic_search=debug,tower_http=debug"))
        .init();

    let state = AppState {
        index: Arc::new(RwLock::new(HnswIndex::new())),
    };

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::info!("semantic search server listening on 0.0.0.0:3000");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::{Request, StatusCode}, Router};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    async fn json_request(router: Router, path: &str, body: Value) -> (StatusCode, Value) {
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[test]
    fn insert_and_search_find_similar_vectors() {
        let mut index = HnswIndex::new();
        index.insert("cat".to_string(), vec![1.0, 0.0, 0.0]);
        index.insert("dog".to_string(), vec![0.0, 1.0, 0.0]);
        index.insert("automobile".to_string(), vec![1.0, 0.1, 0.0]);

        let results = index.search(&[0.95, 0.05, 0.0], 3);
        let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"cat"));
        assert!(ids.contains(&"automobile"));
        assert!(ids.contains(&"dog"));
        assert!(ids[0] != "dog");
        assert!(ids[1] != "dog");
    }

    #[tokio::test]
    async fn http_endpoints_return_expected_payloads() {
        let state = AppState {
            index: Arc::new(RwLock::new(HnswIndex::new())),
        };
        let router = app(state);

        let (status, body) = json_request(
            router.clone(),
            "/index",
            json!({"id": "doc_1", "vector": [0.1, 0.8, 0.2]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], "doc_1");
        assert_eq!(body["total_vectors"], 1);

        let (status, body) = json_request(
            router,
            "/search",
            json!({"query": [0.12, 0.79, 0.18], "top_k": 3}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["results"].is_array());
        assert_eq!(body["results"][0]["id"], "doc_1");
    }
}

"""Bonus Challenge — Hybrid Memory Agent for Vietnamese AI Assistant.

Combines Episodic Memory (Qdrant Vector Store + FastEmbed) with Stable User Profile
and Recent Activity (Feast Feature Store) into a personalized retrieval-augmented context.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = ROOT / "app" / "feast_repo"


class HybridMemoryAgent:
    """Personalized AI assistant agent with hybrid memory architecture."""

    def __init__(self, collection_name: str = "user_episodic_memory") -> None:
        self.collection_name = collection_name
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(":memory:")
        self.memories: list[dict[str, Any]] = []
        self._point_id = 0

        # Initialize Qdrant collection for episodic memories
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Initialize Feast feature store for stable profile & recent activity
        self.fs: FeatureStore | None = None
        if (FEAST_DIR / "registry.db").exists():
            try:
                self.fs = FeatureStore(repo_path=str(FEAST_DIR))
            except Exception:
                self.fs = None

    def remember(self, text: str, user_id: str = "u_001", metadata: dict[str, Any] | None = None) -> None:
        """Store a new piece of episodic memory for a specific user."""
        meta = metadata or {}
        doc = {
            "id": self._point_id,
            "user_id": user_id,
            "text": text,
            "topic": meta.get("topic", "general"),
            "timestamp": meta.get("timestamp", "now"),
        }
        self.memories.append(doc)

        vector = next(self.embedder.embed([text])).tolist()
        point = PointStruct(
            id=self._point_id,
            vector=vector,
            payload={"user_id": user_id, "text": text, "topic": doc["topic"]},
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        self._point_id += 1

    def _search_episodic_hybrid(self, query: str, user_id: str, top_k: int = 3, rrf_k: int = 60) -> list[str]:
        """Hybrid search (BM25 + Vector + RRF) filtered by user_id."""
        user_docs = [m for m in self.memories if m["user_id"] == user_id]
        if not user_docs:
            return []

        # 1. BM25 on user memories
        tokenized = [m["text"].lower().split() for m in user_docs]
        bm25 = BM25Okapi(tokenized)
        kw_scores = bm25.get_scores(query.lower().split())
        kw_ranked_idx = sorted(range(len(kw_scores)), key=lambda i: -kw_scores[i])
        kw_hits = [user_docs[i]["text"] for i in kw_ranked_idx]

        # 2. Vector search on user memories in Qdrant with payload filter
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=user_filter,
            limit=len(user_docs),
        )
        sem_hits = [p.payload["text"] for p in res.points]

        # 3. Reciprocal Rank Fusion (rank 1-based, k=60)
        rrf_scores: dict[str, float] = {}
        for rank, text in enumerate(kw_hits, start=1):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (rrf_k + rank)
        for rank, text in enumerate(sem_hits, start=1):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda kv: -kv[1])
        return [text for text, _ in ranked[:top_k]]

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve user profile, recent activity velocity, and episodic memories into context."""
        # 1. Retrieve profile & velocity from Feast online store
        profile_str = f"User: {user_id}"
        if self.fs:
            try:
                features = self.fs.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:preferred_language",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour",
                        "query_velocity_features:distinct_topics_24h",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                reading_speed = features.get("reading_speed_wpm", [220])[0]
                lang = features.get("preferred_language", ["vi"])[0]
                affinity = features.get("topic_affinity", ["cloud"])[0]
                queries_1h = features.get("queries_last_hour", [5])[0]
                topics_24h = features.get("distinct_topics_24h", [3])[0]

                profile_str = (
                    f"User Profile: [Language: {lang}, Topic Affinity: {affinity}, Reading Speed: {reading_speed} wpm]\n"
                    f"Recent Activity: [{queries_1h} queries in last hour across {topics_24h} topics]"
                )
            except Exception:
                profile_str = f"User Profile: [Default profile for {user_id}]"

        # 2. Hybrid search episodic memories
        top_memories = self._search_episodic_hybrid(query, user_id, top_k=top_k)
        memory_lines = "\n".join([f"  - ({i+1}) {m}" for i, m in enumerate(top_memories)]) if top_memories else "  - (No relevant memories found)"

        # 3. Assemble personalized context
        assembled = (
            f"=== ASSEMBLED CONTEXT FOR ASSISTANT ===\n"
            f"{profile_str}\n"
            f"Episodic Memories (Top-{len(top_memories)}):\n"
            f"{memory_lines}\n"
            f"Current Query: {query}\n"
            f"========================================"
        )
        return assembled

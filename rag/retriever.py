"""
Query-time retrieval: embeds the user's question, searches the
active repository's ChromaDB collection, and returns Top-K chunks
with similarity scores, applying an optional relevance threshold.
"""

from dataclasses import dataclass
from typing import List

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    file_path: str
    file_type: str
    start_line: int
    end_line: int
    content: str
    relevance: float  # 0..1, higher is more relevant


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.query(collection_name, query_embedding, top_k=top_k)

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved: List[RetrievedChunk] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            # Cosine distance -> similarity in [0, 1] (clamped).
            relevance = max(0.0, min(1.0, 1.0 - distance / 2.0))
            if relevance < similarity_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    file_path=meta.get("file_path", "unknown"),
                    file_type=meta.get("file_type", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    content=doc,
                    relevance=relevance,
                )
            )

        return retrieved

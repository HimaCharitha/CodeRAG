"""
ChromaDB wrapper.

Each analyzed repository gets its own collection (named via
utils.repository_utils.make_collection_name), so chunks from
different repositories are never mixed and re-analyzing the same
repo/commit reuses the existing collection instead of re-embedding.
"""

import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from rag.code_chunker import CodeChunk
from rag.embeddings import EmbeddingModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma")


class VectorStore:
    def __init__(self, persist_directory: str = DATA_DIR):
        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

    def collection_exists(self, collection_name: str) -> bool:
        try:
            existing = {c.name for c in self._client.list_collections()}
        except Exception:
            existing = set()
        return collection_name in existing

    def get_or_create_collection(self, collection_name: str):
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, collection_name: str):
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[CodeChunk],
        embedding_model: EmbeddingModel,
        batch_size: int = 64,
        progress_callback=None,
    ):
        collection = self.get_or_create_collection(collection_name)

        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c.content for c in batch]
            embeddings = embedding_model.embed_texts(texts)

            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.to_metadata() for c in batch],
            )

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        return collection

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Optional[dict]:
        if not self.collection_exists(collection_name):
            return None

        collection = self.get_or_create_collection(collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self, collection_name: str) -> int:
        if not self.collection_exists(collection_name):
            return 0
        return self.get_or_create_collection(collection_name).count()

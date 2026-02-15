import faiss
import numpy as np


class FaissIndex:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.node_ids = []

    def add(self, node_id: str, embedding: np.ndarray):
        embedding = embedding.astype("float32")
        embedding = embedding / np.linalg.norm(embedding)
        self.index.add(np.array([embedding]))
        self.node_ids.append(node_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.astype("float32")
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(
            np.array([query_embedding]), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.node_ids):
                results.append((self.node_ids[idx], float(score)))

        return results

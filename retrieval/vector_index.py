import faiss
import numpy as np


class VectorIndex:

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = []

    def add(self, node_id: str, vector: list):
        vec = np.array([vector]).astype("float32")
        self.index.add(vec)
        self.id_map.append(node_id)

    def search(self, query_vector: list, top_k: int):
        vec = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(vec, top_k)
        return [self.id_map[i] for i in indices[0]]

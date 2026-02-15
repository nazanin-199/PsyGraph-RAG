from pathlib import Path
import pandas as pd
import numpy as np

from config import MODEL_NAME, EMBEDDING_MODEL, TOP_K_NODES, MAX_HOPS
from config.settings import Settings
from services.llm_client import LLMClient

from preprocessing.master_processor import MasterPreprocessor
from extraction.llm_extractor import LLMExtractor
from embedding.embedder import Embedder
from embedding.vector_index import FaissIndex
from graph.typed_graph import TypedKnowledgeGraph
from retrieval.graph_retriever import GraphRetriever
from generation.graph_reasoner import GraphReasoner


def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(path)

    required_columns = ["video_id", "transcript"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["transcript"])
    df = df[df["transcript"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    return df


class GraphBuilder:
    def __init__(self, extractor: LLMExtractor, embedder: Embedder):
        self.extractor = extractor
        self.embedder = embedder
        self.graph = TypedKnowledgeGraph()
        self.embedding_dim = None
        self.vector_index = None

    def process_text(self, text: str):
        structured = self.extractor.extract(text)

        for category, values in structured.entities.dict().items():
            for label in values:

                embedding = self.embedder.embed(label)

                if self.embedding_dim is None:
                    self.embedding_dim = len(embedding)
                    self.vector_index = FaissIndex(self.embedding_dim)

                node_id = f"{label}_{category}"

                self.graph.graph.add_node(
                    node_id,
                    label=label,
                    type=category,
                    embedding=embedding,
                )

                self.vector_index.add(node_id, np.array(embedding))

        for rel in structured.relations:
            self.graph.graph.add_edge(
                rel.source,
                rel.target,
                relation=rel.relation,
            )

    def build_from_chunks(self, chunks):
        for chunk in chunks:
            self.process_text(chunk["text"])

        return self.graph, self.vector_index


def run_full_pipeline(csv_path: str, query: str):

    settings = Settings()
    settings.validate()

    llm_client = LLMClient(settings.openrouter_api_key)

    df = load_dataset(csv_path)

    preprocessor = MasterPreprocessor(settings)
    processed_df = preprocessor.process_batch(df)

    chunks = []
    for video in processed_df.to_dict(orient="records"):
        for chunk in video["chunks"]:
            chunks.append(chunk)

    extractor = LLMExtractor(MODEL_NAME)
    embedder = Embedder(EMBEDDING_MODEL)

    builder = GraphBuilder(extractor, embedder)
    graph, vector_index = builder.build_from_chunks(chunks)

    retriever = GraphRetriever(graph, vector_index)

    query_embedding = embedder.embed(query)

    seed_nodes = retriever.retrieve_seed_nodes(
        query_embedding,
        TOP_K_NODES
    )

    expanded_nodes = retriever.multi_hop_expand(
        seed_nodes,
        MAX_HOPS
    )

    subgraph_text = retriever.build_subgraph_text(expanded_nodes)

    reasoner = GraphReasoner(llm_client, MODEL_NAME)

    answer = reasoner.generate_answer(query, subgraph_text)

    return answer


if __name__ == "__main__":

    result = run_full_pipeline(
        csv_path="youtube_videos_export.csv",
        query="How can I improve my self-esteem?"
    )

    print(result)

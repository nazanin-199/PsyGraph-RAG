from config import *
from extraction.llm_extractor import LLMExtractor
from embedding.embedder import Embedder
from graph.graph_store import GraphStore
from graph.node import Node
from graph.edge import Edge
from retrieval.vector_index import VectorIndex
from retrieval.subgraph_retriever import SubgraphRetriever
from retrieval.reasoning import ReasoningEngine
from generation.answer_generator import AnswerGenerator
import uuid


def build_graph_from_text(text, graph, extractor, embedder, vector_index):
    result = extractor.extract(text)

    for category, values in result.entities.dict().items():
        for value in values:
            node_id = str(uuid.uuid4())
            embedding = embedder.embed(value)

            node = Node(
                id=node_id,
                label=value,
                type=category,
                embedding=embedding
            )

            graph.add_node(node)
            vector_index.add(node_id, embedding)

    for rel in result.relations:
        edge = Edge(
            source=rel.source,
            relation=rel.relation,
            target=rel.target,
            confidence=rel.confidence
        )
        graph.add_edge(edge)


def answer_question(question, graph, embedder, vector_index):
    query_vector = embedder.embed(question)
    seed_nodes = vector_index.search(query_vector, TOP_K_NODES)

    subgraph = SubgraphRetriever(graph, MAX_HOPS).retrieve(seed_nodes)
    reasoning_chains = ReasoningEngine(graph).build_reasoning_chain(subgraph)

    generator = AnswerGenerator(MODEL_NAME)
    return generator.generate(question, reasoning_chains)


if __name__ == "__main__":
    graph = GraphStore()
    extractor = LLMExtractor(MODEL_NAME)
    embedder = Embedder(EMBEDDING_MODEL)
    vector_index = VectorIndex(dim=1536)

    sample_text = "User transcript here"

    build_graph_from_text(sample_text, graph, extractor, embedder, vector_index)

    question = "Why do I feel low self esteem?"
    answer = answer_question(question, graph, embedder, vector_index)

    print(answer)

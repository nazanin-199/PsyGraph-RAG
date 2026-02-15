class GraphReasoner:
    def __init__(self, llm_client, model):
        self.llm = llm_client
        self.model = model

    def generate_answer(self, query, subgraph_text):
        prompt = f"""
You are an assistant for psychological support.

Use ONLY the knowledge provided below.

Knowledge Graph:
{subgraph_text}

Question:
{query}

First provide a structured reasoning chain.
Then provide the final answer.
"""

        return self.llm.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

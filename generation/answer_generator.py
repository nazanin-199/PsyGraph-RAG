from openai import OpenAI

client = OpenAI()


class AnswerGenerator:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, question: str, reasoning_chains: list):
        graph_context = "\n".join(reasoning_chains)

        prompt = f"""
        Use the graph reasoning context to answer the user question.
        Base the answer only on provided knowledge.
        Do not invent new psychological claims.

        Question:
        {question}

        Graph Context:
        {graph_context}
        """

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

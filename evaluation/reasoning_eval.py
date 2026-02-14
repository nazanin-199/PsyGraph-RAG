from openai import OpenAI

client = OpenAI()


class ReasoningEvaluator:

    def __init__(self, model_name):
        self.model_name = model_name

    def evaluate_reasoning(self, reasoning_chain, question):
        prompt = f"""
        Evaluate whether the reasoning chain logically supports answering the question.
        Score from 0 to 1.

        Question:
        {question}

        Reasoning Chain:
        {reasoning_chain}
        """

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content

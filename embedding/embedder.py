from openai import OpenAI

client = OpenAI()


class Embedder:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, text: str) -> list:
        response = client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding

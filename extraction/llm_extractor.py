import json
from openai import OpenAI
from .schema import ExtractionResult

client = OpenAI()


class LLMExtractor:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract(self, text: str) -> ExtractionResult:
        prompt = f"""
        Extract structured psychological entities and relations from the text.
        Use relation types only from:
        CAUSES
        CONTRIBUTED_BY
        TREATED_BY
        RELATED_TO

        Return JSON:
        {{
            "entities": {{
                "problems": [],
                "symptoms": [],
                "concepts": [],
                "advice": [],
                "context_factors": []
            }},
            "relations": [
                {{
                    "source": "",
                    "relation": "",
                    "target": "",
                    "confidence": 0.0
                }}
            ]
        }}

        Text:
        {text}
        """

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return ExtractionResult(**data)

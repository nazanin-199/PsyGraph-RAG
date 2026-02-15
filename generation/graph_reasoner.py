"""
Answer generation using knowledge graph context.
"""
from typing import Optional
import logging
from openai import OpenAI, APIError

from exceptions import GenerationError

logger = logging.getLogger(__name__)


class GraphReasoner:
    """
    Generates answers using knowledge graph context and LLM reasoning.
    
    Features:
    - Context truncation for long subgraphs
    - Structured prompting
    - Error handling
    """
    
    def __init__(
        self,
        llm_model: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize reasoner.
        
        Args:
            llm_model: OpenAI model name
            api_key: OpenAI API key
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key)
        self.model = llm_model
        self.timeout = timeout
        self._generation_count = 0
        self._error_count = 0
    
    def generate_answer(
        self,
        query: str,
        subgraph_text: str,
        max_context_length: int = 6000
    ) -> str:
        """
        Generate answer with truncation and structured prompting.
        
        Args:
            query: User question
            subgraph_text: Knowledge graph context
            max_context_length: Maximum context characters
            
        Returns:
            Generated answer
            
        Raises:
            GenerationError: If generation fails
        """
        # Truncate subgraph if too long
        if len(subgraph_text) > max_context_length:
            logger.warning(
                f"Subgraph too long ({len(subgraph_text)} chars), truncating to {max_context_length}"
            )
            subgraph_text = subgraph_text[:max_context_length] + "\n... [truncated for length]"
        
        prompt = f"""You are a psychological support assistant. Answer the user's question using ONLY the knowledge graph provided below.

Knowledge Graph:
{subgraph_text}

Question: {query}

Instructions:
1. If the knowledge graph contains relevant information, provide a clear, helpful answer
2. If the knowledge graph lacks sufficient information, say "I don't have enough information in my knowledge base to answer this question fully"
3. Structure your response as:
   - Reasoning: Briefly explain your thought process based on the knowledge graph
   - Answer: Provide the final answer
4. Cite specific entities or relations from the knowledge graph when applicable

Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                timeout=self.timeout
            )
            
            answer = response.choices[0].message.content
            self._generation_count += 1
            
            logger.info(f"Generated answer of length {len(answer)}")
            return answer
            
        except APIError as e:
            self._error_count += 1
            logger.error(f"Answer generation API error: {e}")
            raise GenerationError(f"Failed to generate answer: {e}") from e
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Unexpected error during generation: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def get_stats(self) -> dict:
        """Return generation statistics."""
        return {
            'total_generations': self._generation_count,
            'total_errors': self._error_count,
            'error_rate': self._error_count / max(self._generation_count + self._error_count, 1)
        }

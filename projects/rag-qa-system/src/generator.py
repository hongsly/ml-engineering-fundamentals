from openai import OpenAI, OpenAIError
from src.utils import Chunk

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a Q&A assistant for RAG (retrieval-augmented generation) research. "
    + "Answer ONLY using information from the provided support material. "
    + "When citing information, reference the source document. "
    + "If the support material does not contain enough information to answer the question, respond with: "
    + "'I don't have enough information in the provided materials to answer this question. '"
    + "DO NOT use your general knowledge - only cite the support material. "
)

SYSTEM_PROMPT_WITHOUT_CONTEXT = (
    "You are a Q&A assistant for RAG (retrieval-augmented generation) research. "
    + "Answer the user's question in RAG area. "
)


class Generator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(
        self, query: str, context: list[Chunk] | None = None, model: str = "gpt-4o-mini"
    ) -> str:
        instructions = (
            SYSTEM_PROMPT_WITH_CONTEXT if context else SYSTEM_PROMPT_WITHOUT_CONTEXT
        )
        try:
            response = self.client.responses.create(
                model=model,
                input=self._get_prompt(query, context),
                instructions=instructions,
            )
            print(f"Tokens used: {response.usage}")
            return response.output_text
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _get_prompt(self, query: str, context: list[Chunk] | None = None):
        prompt = f"<question>{query}</question>"
        if context:
            prompt += "\n<documents>\n"
            doc_strings = [
                f'<document id="{c["chunk_id"]}">{c["chunk_text"]}</document>'
                for c in context
            ]
            prompt += "\n".join(doc_strings)
            prompt += "\n</documents>"
        return prompt

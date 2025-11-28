from openai import OpenAI, OpenAIError
from src.utils import Chunk

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a Q&A assistant for RAG (retrieval-augmented generation) research. "
    + "Answer ONLY using information from the provided support material in <documents>...</documents>. "
    + "When citing information, ALWAYS include reference to the source <document> using its <metadata>. "
    + "If the support <documents> do not contain enough information to answer the question, respond with: "
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
        self, query: str, context: list[Chunk] | None = None, model: str = "gpt-4o-mini", retrieval_mode: str = "none"
    ) -> str:
        instructions = (
            SYSTEM_PROMPT_WITH_CONTEXT if context else SYSTEM_PROMPT_WITHOUT_CONTEXT
        )
        try:
            response = self.client.responses.create(
                model=model,
                input=self._get_prompt(query, context),
                instructions=instructions,
                metadata={"retrieval_mode": retrieval_mode}
            )
            print(f"Tokens used: {response.usage}")
            return response.output_text
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _get_prompt(self, query: str, context: list[Chunk] | None = None):
        prompt = f"<question>{query}</question>"
        if context:
            prompt += "\n<documents>\n"
            for c in context:
                prompt += f' <document id="{c["chunk_id"]}">\n'
                prompt += "  <metadata>\n"
                prompt += f'   <title>{c["metadata"]["title"]}</title>\n'
                prompt += f'   <authors>{",".join(c["metadata"]["authors"])}</authors>\n'
                prompt += f'   <year>{c["metadata"]["year"]}</year>\n'
                prompt += "  </metadata>\n"
                prompt += f'  <content>{c["chunk_text"]}</content>\n'
                prompt += " </document>\n"
            prompt += "</documents>"
        return prompt

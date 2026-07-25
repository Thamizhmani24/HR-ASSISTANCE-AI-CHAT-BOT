from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "xxxxxxxx" in api_key or api_key == "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
            raise ValueError(
                "Invalid or missing OPENAI_API_KEY in your .env file. "
                "Please replace the placeholder key with your actual OpenAI API key."
            )
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            _client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            _client = OpenAI(api_key=api_key)
    return _client


def query_llm_with_context(query: str, context: str):
    system_content = """You are a helpful assistant for answering user queries based on provided context. 
    use the context to provide accurate and relevant answers. Do not make assumptions beyond the context provided.
    If the context does not contain enough information to answer the query, 
    let the user know that you cannot provide an answer based on the given context.
    """
    response = get_openai_client().chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content


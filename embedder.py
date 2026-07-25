from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables from .env file
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


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")



def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embeds chunks using OpenAI's embedding model."""
    embeddings = []
    for chunk in chunks:
        response = get_openai_client().embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
        )
        embeddings.append(response.data[0].embedding)

    return embeddings


def embed_User_query(query: str) -> List[float]:
    """Embeds a user query using OpenAI's embedding model."""
    response = get_openai_client().embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding




# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# print("API Key:", os.getenv("OPENAI_API_KEY"))

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
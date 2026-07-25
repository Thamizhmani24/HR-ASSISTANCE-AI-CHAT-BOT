from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from typing import List
import time

# Load environment variables from .env file
load_dotenv()

_index = None

def get_index(dimension: int = 1536):
    global _index
    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not api_key or "****************" in api_key or api_key == "pcsk_************************":
            raise ValueError(
                "Invalid or missing PINECONE_API_KEY in your .env file. "
                "Please replace the placeholder key with your actual Pinecone API key."
            )
        if not index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME is not set in environment variables. "
                "Please configure it in the .env file."
            )
            
        pinecone_client = Pinecone(api_key=api_key)
        
        # Check if the index exists and manage dimensions
        try:
            existing_indexes = pinecone_client.list_indexes().names()
        except AttributeError:
            # Fallback if list_indexes doesn't have .names() in different package versions
            existing_indexes = [idx.name for idx in pinecone_client.list_indexes()]

        if index_name not in existing_indexes:
            print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be initialized
            while not pinecone_client.describe_index(index_name).status.ready:
                time.sleep(1)
        else:
            desc = pinecone_client.describe_index(index_name)
            if desc.dimension != dimension:
                print(f"Dimension mismatch for index '{index_name}': expected {dimension}, got {desc.dimension}.")
                print(f"Recreating index '{index_name}' with dimension {dimension}...")
                pinecone_client.delete_index(index_name)
                pinecone_client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                while not pinecone_client.describe_index(index_name).status.ready:
                    time.sleep(1)
            
        _index = pinecone_client.Index(index_name)
    return _index


def store_in_pinecone(chunks: List[str], embeddings: List[List[float]], namespace: str = ""):
    if not embeddings:
        return
    dimension = len(embeddings[0])
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_data = {
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i
            }
        }
        vectors_to_upsert.append(vector_data)
    
    # Upsert vectors in batches (Pinecone recommends batch size of 100)
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        get_index(dimension).upsert(vectors=batch, namespace=namespace)


def search_in_pinecone(query_vector: List[float], top_k: int = 1, namespace: str = ""):
    dimension = len(query_vector)
    results = get_index(dimension).query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    print(f"Found {len(results.matches)} matches for the query.")
    matched_chunks = []
    for match in results.matches:
        matched_chunks.append(match.metadata.get("text", ""))
    return matched_chunks


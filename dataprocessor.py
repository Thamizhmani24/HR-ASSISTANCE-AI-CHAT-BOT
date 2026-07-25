from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone
from typing import List

def run(target_pdf_path: str = "./resources/HRPolicy.pdf"):
    pages = read_pdf(target_pdf_path)
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)
    embedded_chunks = embed_chunks(chunks)
    print(f"Total chunks created: {len(embedded_chunks)}")
    if embedded_chunks:
        print(f"First chunk embedding dimension: {len(embedded_chunks[0])}")
        store_in_pinecone(chunks, embedded_chunks, namespace="")
    return len(embedded_chunks)

   
    
if __name__ == "__main__":
    run()
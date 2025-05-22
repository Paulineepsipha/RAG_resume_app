import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def retrieve_relevant_chunks(query: str, collection_name="rag_resume", top_k=3):
    client = chromadb.Client(chromadb.config.Settings(
        persist_directory="./chromadb_persist"
    ))
    collection = client.get_collection(name=collection_name)

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    query_embedding = embedding_fn(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results['documents'][0]

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        chunks = retrieve_relevant_chunks(user_query)
        print("\nRelevant chunks:\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:\n{chunk}\n")

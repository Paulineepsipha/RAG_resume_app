# import chromadb
# # from chromadb.config import Settings
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# persist_dir = "chroma_db"
# collection_name = "rag_resume"

# def retrieve_relevant_chunks(query, k=3):
#     client = chromadb.PersistentClient(path=persist_dir)

#     embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

#     try:
#         collection = client.get_collection(name=collection_name, embedding_function=embedding_fn)
#     except Exception as e:
#         print(f"Error loading collection: {e}")
#         return []

#     results = collection.query(
#         query_texts=[query],
#         n_results=k,
#         include=["documents", "distances"]
#     )

#     documents = results["documents"][0]  # list of top-k matched chunks
#     distances = results["distances"][0]  # similarity distances (lower = more similar)


#     # Combine text with similarity score
#     for i, (doc, dist) in enumerate(zip(documents, distances)):
#         print(f"\nChunk {i+1} (Score: {1 - dist:.4f} similarity):\n{doc}")

#     return documents

# if __name__ == "__main__":
#     while True:
#         user_query = input("Enter your query (or 'exit' to quit): ")
#         if user_query.lower() == 'exit':
#             break
        
#         chunks = retrieve_relevant_chunks(user_query)
#         # if chunks:
#         #     print("\nTop Relevant Chunks:")
#         #     for i, chunk in enumerate(chunks, 1):
#         #         print(f"\nChunk {i}: {chunk}")
#         # else:
#         #     print("No results found.")



## the below code is for qa_engine Integration Plan:
# retriever_chromadb.py

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

persist_dir = "chroma_db"
collection_name = "rag_resume"

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=persist_dir)


# Get or create collection safely
existing_collections = [col.name for col in client.list_collections()]
if collection_name in existing_collections:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)


def retrieve_relevant_chunks(query: str, top_k: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    documents = results["documents"][0]
    # scores = results["distances"][0]  # optional
    return documents  # optionally return scores too

# Standalone test loop
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        chunks = retrieve_relevant_chunks(user_query)
        print("\nTop matching chunks:\n")
        for i, chunk in enumerate(chunks):
            print(f"{i+1}. {chunk}\n")

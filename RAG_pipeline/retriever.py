import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Step 1: Load saved embeddings and text chunks
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

text_chunks = data["text_chunks"]
embeddings = np.array(data["embeddings"])

# Step 2: Load the embedding model (same one used earlier)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Define a function to retrieve top-k relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(text_chunks, convert_to_tensor=True)

    # Use cosine similarity
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # semantic_search returns a list of lists

    print(f"\nTop {top_k} relevant chunks for query: \"{query}\"")
    for i, hit in enumerate(hits):
        score = hit['score']
        chunk = text_chunks[hit['corpus_id']]
        print(f"\n--- Chunk {i+1} (Score: {score:.4f}) ---\n{chunk}\n")

# Example test
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        retrieve_relevant_chunks(user_query)

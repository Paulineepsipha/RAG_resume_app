import requests
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Load saved embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

text_chunks = data["text_chunks"]
embeddings = np.array(data["embeddings"])

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embedding-based chunk retrieval
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(text_chunks, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    return [text_chunks[hit["corpus_id"]] for hit in hits[0]]

# Call DeepSeek API
def call_deepseek_api(prompt, model="deepseek-chat"):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API call failed: {e}")
        return "Error: Could not get response from DeepSeek."

# Generate an answer using RAG
def generate_answer(query, top_k=3):
    relevant_chunks = retrieve_relevant_chunks(query, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

    return call_deepseek_api(prompt)

# === INTERFACE ===
if __name__ == "__main__":
    while True:
        user_query = input("Ask a question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = generate_answer(user_query)
        print("\nAnswer:\n", answer, "\n")
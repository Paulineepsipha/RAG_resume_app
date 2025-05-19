import requests
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import json

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

#  Call Ollama API and stream the response correctly
def call_ollama(prompt, model_name="tinyllama"):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            },
            stream=True
        )
        response.raise_for_status()

        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content", "")
                    answer += content
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
        return answer.strip()

    except requests.exceptions.RequestException as e:
        return f"Ollama request failed: {e}"


# Generate an answer using RAG
def generate_answer(query, top_k=3):
    relevant_chunks = retrieve_relevant_chunks(query, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

    response = call_ollama(prompt)
    return response if response else "No response generated."

# === INTERFACE ===
if __name__ == "__main__":
    while True:
        user_query = input("Ask a question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = generate_answer(user_query)
        print("\nAnswer:\n", answer, "\n")

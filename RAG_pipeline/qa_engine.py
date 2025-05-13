import openai
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

# Set your OpenAI API key (replace with your actual key)
openai.api_key = " Own_API" 

# Load saved embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

text_chunks = data["text_chunks"]
embeddings = np.array(data["embeddings"])

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(text_chunks, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    return [text_chunks[hit["corpus_id"]] for hit in hits[0]]

def generate_answer(query, top_k=3):
    relevant_chunks = retrieve_relevant_chunks(query, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use "gpt-4" if you have access to it
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    return response['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    while True:
        user_query = input("Ask a question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = generate_answer(user_query)
        print("\nAnswer:\n", answer, "\n")

import json
import os
import pickle

# Assuming you have already loaded text_chunks and embeddings from your previous steps

# Example dummy data
text_chunks = [
    "This is the first chunk of text.",
    "This is the second chunk of text."
]

# Assuming `embeddings` are the embeddings you got from the model (e.g., SentenceTransformer)
embeddings = [
    [0.1, 0.2, 0.3],  # Example embeddings
    [0.4, 0.5, 0.6]
]

# Option 1: Saving as a Pickle file (you can later load it for retrieval)
def save_embeddings_pickle(text_chunks, embeddings, filename="embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"text_chunks": text_chunks, "embeddings": embeddings}, f)
    print(f"Embeddings saved to {filename}.")

# Option 2: Saving as a JSON file (if you prefer JSON format)
def save_embeddings_json(text_chunks, embeddings, filename="embeddings.json"):
    # Convert embeddings to a list of lists format suitable for JSON
    data = {"text_chunks": text_chunks, "embeddings": embeddings}
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Embeddings saved to {filename}.")

# Choose the format you want to use
save_embeddings_pickle(text_chunks, embeddings)  # Or use `save_embeddings_json`

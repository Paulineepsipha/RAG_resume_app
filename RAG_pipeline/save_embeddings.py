# save_embeddings.py
import pickle

def save_embeddings_pickle(text_chunks, embeddings, filename="embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"text_chunks": text_chunks, "embeddings": embeddings}, f)
    print(f"Embeddings saved to {filename}")

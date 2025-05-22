import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

persist_dir = "chroma_db"

def save_to_chromadb(text_chunks, collection_name="rag_resume"):
    client = chromadb.Client(persist_directory=persist_dir)

    # Optional: Clear existing collection
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)

    # Use the same embedder as before
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)

    for i, chunk in enumerate(text_chunks):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

    print(f"Saved {len(text_chunks)} chunks to ChromaDB under collection '{collection_name}'")


if __name__ == "__main__":
    
    # load and chunk your document here
    from loader import load_pdf
    from chunker import chunk_text

    text = load_pdf(r"C:\Users\melvi\Desktop\MONASH STUDY\Personal_project\Ragmodel_chatbot\CODE\RAG_resume_app\Data\monopoly.pdf")
    chunks = chunk_text(text)

    save_to_chromadb(chunks)
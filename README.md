# RAG_resume_app
An AI-powered web application that uses Retrieval-Augmented Generation (RAG) to answer natural language questions about uploaded resumes. Built with Python, sentence-tranformer (hugging face), and Streamlit.


# RAG Resume Chatbot — Embeddings & Vector Storage
This project demonstrates two approaches to embedding resume text chunks and saving embeddings for retrieval in a Retrieval-Augmented Generation (RAG) chatbot:

## 1. Embedding & Saving with Pickle (Manual)
  Extracts text from PDF → splits text into chunks → generates embeddings using SentenceTransformer → saves embeddings and chunks as a pickle file.
  
  Good for understanding the embedding pipeline and saving vectors offline.
    Files involved:
    loader.py — loads PDF text
    chunker.py — splits text into chunks
    embedder.py — generates embeddings from chunks
    save_embeddings.py — saves chunks + embeddings as a pickle file

## 2. Embedding & Saving with ChromaDB (Automated)
  Extracts and chunks text, then saves text chunks to ChromaDB vector store directly, using Chroma’s integration with SentenceTransformer to embed chunks automatically.
  
  Recommended for scalable vector search and real-time retrieval.
  Files involved:
    loader.py — loads PDF text
    chunker.py — splits text into chunks
    save_to_chromadb.py — saves chunks to ChromaDB with embedding function

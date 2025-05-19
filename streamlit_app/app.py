import streamlit as st
import sys
import os

# Add RAG_pipeline to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG_pipeline")))


from retriever import retrieve_relevant_chunks

st.set_page_config(page_title="RAG Chunk Retriever", layout="centered")
st.title("📄 Ask your PDF")
st.write("Enter a question to retrieve the most relevant text chunks from the document.")

query = st.text_input("Enter your query:")

if query:
    with st.spinner("Retrieving relevant chunks..."):
        top_chunks = retrieve_relevant_chunks(query, top_k=3)

    st.markdown("### 🔍 Top Matching Chunks")
    for i, chunk in enumerate(top_chunks, 1):
        st.markdown(f"**Chunk {i}:**")
        st.write(chunk)
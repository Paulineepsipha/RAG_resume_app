# from RAG_pipeline.loader import load_pdf
# from RAG_pipeline.chunker import chunk_text
# from RAG_pipeline.embedder import embed_chunks

# # 1. Load PDF
# text = load_pdf(r"C:\Users\melvi\Desktop\MONASH STUDY\Personal_project\Ragmodel_chatbot\CODE\RAG_resume_app\Data\dgsaustralia_data.pdf")

# # 2. Chunk it
# chunks = chunk_text(text)

# # 3. Embed the chunks
# embeddings = embed_chunks(chunks)

# # 4. Optional: Check one result
# print(f"First chunk:\n{chunks[0][:200]}...\n")
# print(f"Embedding shape: {embeddings[0].shape}")


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "RAG_pipeline"))


from RAG_pipeline.loader import load_pdf
from RAG_pipeline.chunker import chunk_text
from RAG_pipeline.embedder import embed_chunks
from save_embeddings import save_embeddings_pickle

# 1. Load PDF
text = load_pdf(r"C:\Users\melvi\Desktop\MONASH STUDY\Personal_project\Ragmodel_chatbot\CODE\RAG_resume_app\Data\monopoly.pdf")

# 2. Chunk it
chunks = chunk_text(text)

# 3. Embed it
embeddings = embed_chunks(chunks)

# 4. Save
save_embeddings_pickle(chunks, embeddings)

print(f"Total chunks created: {len(chunks)}")
print(f"Shape of first embedding: {embeddings[0].shape}")


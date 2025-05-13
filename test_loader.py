## Loader.py
# from RAG_pipeline.loader import load_pdf

# text = load_pdf(r"C:\Users\melvi\Desktop\MONASH STUDY\Personal_project\Ragmodel_chatbot\CODE\RAG_resume_app\Data\PaulineEpsipha_dataresume.pdf")
# print(text[:500]) 

# #chunker.py
# from RAG_pipeline.loader import load_pdf
# from RAG_pipeline.chunker import chunk_text

# text = load_pdf(r"C:\Users\melvi\Desktop\MONASH STUDY\Personal_project\Ragmodel_chatbot\CODE\RAG_resume_app\Data\dgsaustralia_data.pdf")
# chunks = chunk_text(text)

# print(f"Number of chunks: {len(chunks)}")
# print("First chunk preview:")
# print(chunks[0])


##embedder.py
from RAG_pipeline.loader import load_pdf
from RAG_pipeline.chunker import chunk_text
from RAG_pipeline.embedder import embed_chunks

# Load and chunk
text = load_pdf(r"C:\Users\melvi\Desktop\MONASH STUDY\Personal_project\Ragmodel_chatbot\CODE\RAG_resume_app\Data\dgsaustralia_data.pdf")
chunks = chunk_text(text)

# Embed
embeddings = embed_chunks(chunks)

print(f"Number of chunks: {len(chunks)}")
print(f"Embedding shape of first chunk: {embeddings[0].shape}")

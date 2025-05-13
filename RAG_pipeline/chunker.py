from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks of words.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

        # Print the first 300 characters of the chunk for inspection
        print(f"Chunk {len(chunks)}: {chunks[-1][:300]}...")

    return chunks

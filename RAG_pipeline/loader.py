import fitz  # PyMuPDF
from pathlib import Path

def load_pdf(file_path):
    """Reads text content from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

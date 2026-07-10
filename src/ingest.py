from pathlib import Path

from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.loader import scan_extract_data

FAISS_FOLDER = Path("faiss_index/")

doc_data = scan_extract_data() # Ensure data is extracted before chunking and indexing

def split_text():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    langchain_style_data = [
        Document(page_content=item["text"], metadata={"source": item["file_path"]}) for item in doc_data
    ]
    chunks = splitter.split_documents(langchain_style_data)
    return chunks

def embed_text():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def create_faiss_index(chunks, embeddings):
    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(str(FAISS_FOLDER))
    return print(f"FAISS index created and saved to {FAISS_FOLDER}")

def ingest_documents():
    if not doc_data:
        print("No documents found to ingest. Please add pdfs or docx files to the 'data' folder.")
        return
    chunks = split_text()
    embeddings = embed_text()
    create_faiss_index(chunks, embeddings)

if __name__ == "__main__":
    ingest_documents()
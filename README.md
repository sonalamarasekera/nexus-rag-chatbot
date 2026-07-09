# NAIVE-RAG Chatbot (HuggingFace Embeddings + FAISS + Ollama)

This is a **Retrieval Augmented Generation (RAG) chatbot** that can handle pdfs and docx files. Users are able to ask questions from documents uploaded to the specified folder (data).

It is built using **Langchain**, **HuggingFace Embeddings**, **FAISS**, **Ollama** and then **Streamlit** for the UI.

---

## Setting up the chatbot
### 1. Create a virtual environment
```bash
# Creating virtual environment
python -m venv venv

# Activate the environment
# --- Windows ---
venv/Scripts/activate
# --- MacOS/Linux ---
source venv/bin/activate
```
### 2. Install dependencies
```bash
# Clone the repository
git clone https://github.com/sonalamarasekera/nexus-rag-chatbot.git

# After moving into the cloned repo, install dependencies
pip install -r requirements.txt

# Also install the additional package below
# --- Windows ---
winget install zstd
# --- MacOS ---
brew install zstd
# --- Linux ---
sudo apt-get install zstd
```
### 3. Build the FAISS index (Vector database)
Place the documents (.pdf and .docx) inside the data folder on the main page, then run the following
```bash
python -m src.ingest
```
This instructs documents to be chunked and embedded, and the FAISS index will be created (2 files in faiss_index folder)

### 4. Download Ollama and preferred model
```bash
# Download from https://ollama.com/download or use:
curl -fsSL https://ollama.com/install.sh | sh

# Keep ollama running (possibly in a separate terminal)
ollama serve

# Pull model of choosing from Ollama ('mistral' for balanced use, 'gemma2:2b' for hardware constraints)
ollama pull mistral
```

### 5. Run UI
After all the above steps are successfully completed, run the following
```bash
streamlit run app.py
```
The chatbot will run on: http://localhost:8501 (or another available port that streamlit chooses)
---
Breakdown of the process:
- Documents are loaded and the content is broken into chunks.
- Each of these chunks are into embeddings and stored in the FAISS vector database
- When user asks a question, FAISS retrieves relevant chunks
- These chunks are passed as context to the LLM
- The LLM genereates accurate answers based on the context
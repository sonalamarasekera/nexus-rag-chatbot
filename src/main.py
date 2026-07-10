# Used to perfrom the RAG function to retrive data from the FAISS database and perform semantic search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from src.ingest import ingest_documents, FAISS_FOLDER

def vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorDB = FAISS.load_local(str(FAISS_FOLDER), embeddings, allow_dangerous_deserialization=True)
    return vectorDB

def init_llm(model:str):
    return OllamaLLM(model=model)

def get_answer(query:str, model:str, k:int = 4):
    data = vector_db()
    
    retriever = data.as_retriever(search_kwargs={"k": k})
    
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f'''
    You are an autonomous HR Assistant Agent. Answer the question using ONLY the provided context below.
    If you do not know the answer, say 'I cannot find that in the database.'\n\n
    Context:{context}
    Question: {query}
    Answer:
    '''

    llm = init_llm(model)

    return llm.invoke(system_prompt).strip()
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def test_rag():
    print("--- Testing RAG Retrieval ---")
    
    if not os.path.exists("faiss_index"):
        print("ERROR: faiss_index not found. Did you run ingest.py?")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("SUCCESS: Vector index loaded.")
        
        # Test Query
        query = "What is the procedure for pipeline pigging?"
        print(f"\nQuery: {query}")
        
        docs = vector_store.similarity_search(query, k=3)
        if docs:
            print(f"\nSUCCESS: Retrieved {len(docs)} chunks.")
            for i, doc in enumerate(docs):
                print(f"--- Chunk {i+1} (Source: {doc.metadata.get('source_file')}, Page: {doc.metadata.get('page')}) ---")
                print(doc.page_content[:150] + "...")
        else:
            print("WARNING: No docs retrieved.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_rag()

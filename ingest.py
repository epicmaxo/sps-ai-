import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

KNOWLEDGE_DIR = "knowledge"
INDEX_PATH = "faiss_index"

def ingest_knowledge_base():
    # 1. Setup
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print(f"Created '{KNOWLEDGE_DIR}' directory. Please put your PDFs here.")
        return

    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{KNOWLEDGE_DIR}'. Please add some manuals.")
        return

    print(f"Found {len(pdf_files)} PDFs in '{KNOWLEDGE_DIR}':")
    for f in pdf_files:
        print(f" - {os.path.basename(f)}")

    # 2. Load Documents
    all_documents = []
    print("\n--- Loading Documents ---")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Enhance metadata
            filename = os.path.basename(pdf_path)
            for doc in docs:
                doc.metadata["source_file"] = filename
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    print(f"Total pages loaded: {len(all_documents)}")

    # 3. Split Text
    print("\n--- Splitting Text ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Create Embeddings & Vector Store
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    print("\n--- Generating Embeddings & Indexing ---")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Process in batches if huge, but for 1000 pages (~100MB), FAISS handles it fine in memory
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 5. Save Index
    vector_store.save_local(INDEX_PATH)
    print(f"\nSUCCESS: Vector store saved to '{INDEX_PATH}'")
    print("Ready for Elite RAG.")

if __name__ == "__main__":
    ingest_knowledge_base()

# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# --- CHANGE 1: Import the new splitter ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker # This is no longer needed

# Import API keys from config
from .config import PINECONE_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Hugging Face embedding model. This is still needed to create the vectors for Pinecone.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
INDEX_NAME = "rag-index"

# --- Retriever (Existing function) ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Correct dimension for 'all-MiniLM-L6-v2'
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()

# --- Function to add documents to the vector store ---
def add_document_to_vectorstore(text_content: str):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text using a recursive character-based method before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    # Ensure the index exists before we try to add documents to it.
    get_retriever()

    # --- CHANGE 2: Replace SemanticChunker with RecursiveCharacterTextSplitter ---
    # This splitter is lightweight and does not require an embedding model to run.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # The target size of each chunk in characters
        chunk_overlap=200,    # The number of characters to overlap between chunks
        add_start_index=True, # Adds the starting character index to metadata
    )
    
    documents = text_splitter.create_documents([text_content])
    
    # --- CHANGE 3: Updated print statement for clarity ---
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    
    # Get the vectorstore instance (not the retriever) to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Add documents to the vector store.
    # The embedding model is used here, not during splitting.
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

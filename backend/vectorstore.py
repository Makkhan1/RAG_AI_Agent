# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Import API keys from config
from .config import PINECONE_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "rag-index"

# --- LAZY LOADING IMPLEMENTATION ---

# 1. Create a global variable to hold the model, initialized to None.
_embeddings_model = None

def get_embeddings():
    """
    Lazily loads the embedding model to save memory on startup.
    It will only be loaded into memory on the first call to this function.
    Subsequent calls will return the already-loaded model.
    """
    global _embeddings_model
    if _embeddings_model is None:
        print("--- LAZY LOADING EMBEDDING MODEL ---")
        # This is the line that uses a lot of RAM, but now it only runs once when needed.
        _embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("--- EMBEDDING MODEL LOADED ---")
    return _embeddings_model

# --- END OF LAZY LOADING IMPLEMENTATION ---


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
    
    # Use the lazy loading function here
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=get_embeddings())
    return vectorstore.as_retriever()


def add_document_to_vectorstore(text_content: str):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into semantically coherent chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    # This line ensures the index exists before we try to add documents to it.
    get_retriever()

    # Use the lazy loading function to get the embeddings model
    embeddings = get_embeddings()
    
    # Use the SemanticChunker with the now-loaded embeddings
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    documents = text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} semantically coherent chunks for indexing...")
    
    # Use the lazy loading function again to initialize the vectorstore for writing
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

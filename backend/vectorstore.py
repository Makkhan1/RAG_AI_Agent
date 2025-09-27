# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import API keys from config
from .config import PINECONE_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index name
INDEX_NAME = "rag-index"

# --- LAZY LOADING IMPLEMENTATION ---
# Global variable to hold the embeddings model instance
_EMBEDDINGS = None

def get_embeddings_model():
    """
    Initializes and returns the HuggingFaceEmbeddings model.
    Uses a global variable to ensure it's only loaded once.
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print("Loading embeddings model for the first time...")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embeddings model loaded.")
    return _EMBEDDINGS
# --- END OF LAZY LOADING IMPLEMENTATION ---


# --- Retriever (Updated function) ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""
    # Ensure the index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Correct dimension for 'all-MiniLM-L6-v2'
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")
    
    # Get the embeddings model using our lazy loader
    embeddings = get_embeddings_model()
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()

# --- Function to add documents to the vector store (Updated function) ---
def add_document_to_vectorstore(text_content: str):
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    # This call ensures the index exists before we try to add documents to it.
    # It will also trigger the model loading on the very first run.
    get_retriever()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        add_start_index=True,
    )
    
    documents = text_splitter.create_documents([text_content])
    
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    
    # Get the embeddings model using our lazy loader
    embeddings = get_embeddings_model()
    
    # Get the vectorstore instance (not the retriever) to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Add documents to the vector store
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

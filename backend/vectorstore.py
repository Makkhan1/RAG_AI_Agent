# # rag_agent_app/backend/vectorstore.py

# import os
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings # Changed to HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Import API keys from config (only Pinecone is needed here now)
# # Import API keys from config (only Pinecone is needed here now)
# from .config import PINECONE_API_KEY

# # Set environment variables for Pinecone
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Define Hugging Face embedding model
# # This will download the model the first time it's used.
# # The default model for HuggingFaceEmbeddings is 'sentence-transformers/all-MiniLM-L6-v2'
# # which has a dimension of 384.
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Define Pinecone index name
# INDEX_NAME = "rag-index" # Make sure this matches your actual index name

# # --- Retriever (Existing function) ---
# def get_retriever():
#     """Initializes and returns the Pinecone vector store retriever."""
#     # Ensure the index exists, create if not
#     if INDEX_NAME not in pc.list_indexes().names():
#         print(f"Creating new Pinecone index: {INDEX_NAME}...")
#         pc.create_index(
#             name=INDEX_NAME,
#             dimension=1024, # Changed dimension for 'sentence-transformers/all-MiniLM-L6-v2'
#             metric="cosine",
#             spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust cloud/region as per your Pinecone setup
#         )
#         print(f"Created new Pinecone index: {INDEX_NAME}")
    
#     vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
#     return vectorstore.as_retriever()

# # --- Function to add documents to the vector store ---
# def add_document_to_vectorstore(text_content: str):
#     """
#     Adds a single text document to the Pinecone vector store.
#     Splits the text into chunks before embedding and upserting.
#     """
#     if not text_content:
#         raise ValueError("Document content cannot be empty.")

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         add_start_index=True,
#     )
    
#     # Create Langchain Document objects from the raw text
#     documents = text_splitter.create_documents([text_content])
    
#     print(f"Splitting document into {len(documents)} chunks for indexing...")
    
#     # Get the vectorstore instance (not the retriever) to add documents
#     vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
#     # Add documents to the vector store
#     vectorstore.add_documents(documents)
#     print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

# rag_agent_app/backend/vectorstore.py

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

# Define Hugging Face embedding model
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
    Splits the text into semantically coherent chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    # --- THIS IS THE FIX ---
    # This line ensures the index exists before we try to add documents to it.
    get_retriever()
    # --- END OF FIX ---

    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    documents = text_splitter.create_documents([text_content])
    
    print(f"Splitting document into {len(documents)} semantically coherent chunks for indexing...")
    
    # Get the vectorstore instance (not the retriever) to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Add documents to the vector store
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
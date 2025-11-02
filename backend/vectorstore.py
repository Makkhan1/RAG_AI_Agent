# rag_agent_app/backend/vectorstore.py

import os
import requests
import json
from typing import List

# FIXED: Correct Pinecone imports with version compatibility
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_V3 = True
    print("Using Pinecone v3.x")
except ImportError:
    try:
        import pinecone
        PINECONE_V3 = False
        print("Using Pinecone v2.x")
    except ImportError:
        raise ImportError("Pinecone library not found. Please install: pip install pinecone-client")

from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# Import the API keys from config
from .config import PINECONE_API_KEY, JINA_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# FIXED: Initialize Pinecone client with proper version handling
if PINECONE_V3:
    # Pinecone v3.x
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    # Pinecone v2.x
    pinecone.init(api_key=PINECONE_API_KEY)
    pc = pinecone

# Define Pinecone index name
INDEX_NAME = "rag-index"

# --- Custom LangChain Embeddings class for Jina AI ---
class JinaApiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "jina-embeddings-v3", task: str = "text-matching"):
        if not api_key:
            raise ValueError("Jina API key not found. Please set the JINA_API_KEY environment variable.")
        self.api_key = api_key
        self.model_name = model_name
        self.task = task
        self.api_url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Internal method to call the Jina API and handle responses."""
        # Filter out empty strings, as they can cause issues with some APIs
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return []

        data = {
            "model": self.model_name,
            "task": self.task,
            "input": valid_texts
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            if "data" not in result or not result["data"]:
                raise ValueError(f"Jina API returned no data. Response: {result}")

            # Sort embeddings by their original index to maintain order
            sorted_embeddings = sorted(result["data"], key=lambda e: e["index"])
            return [item["embedding"] for item in sorted_embeddings]

        except requests.exceptions.RequestException as e:
            print(f"Error calling Jina API: {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"Error parsing Jina API response: {e}")
            # It's helpful to see the problematic response text
            if 'response' in locals():
                print(f"Received response text: {response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents for storage."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query for retrieval."""
        result = self._embed([text])
        return result[0] if result else []

# Initialize our custom Jina embeddings class
embeddings = JinaApiEmbeddings(api_key=JINA_API_KEY)

# FIXED: Helper functions for Pinecone operations
def get_pinecone_index(index_name: str):
    """Get Pinecone index with version compatibility."""
    try:
        if PINECONE_V3:
            # Pinecone v3.x
            return pc.Index(index_name)
        else:
            # Pinecone v2.x
            return pinecone.Index(index_name)
    except Exception as e:
        print(f"Error getting Pinecone index '{index_name}': {e}")
        raise

def create_pinecone_index(index_name: str, dimension: int = 1024):
    """Create Pinecone index with version compatibility."""
    try:
        if PINECONE_V3:
            # Pinecone v3.x
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        else:
            # Pinecone v2.x
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        print(f"Created new Pinecone index: {index_name}")
    except Exception as e:
        print(f"Error creating Pinecone index '{index_name}': {e}")
        raise

def list_pinecone_indexes():
    """List Pinecone indexes with version compatibility."""
    try:
        if PINECONE_V3:
            # Pinecone v3.x
            indexes = pc.list_indexes()
            if hasattr(indexes, 'names'):
                return indexes.names()
            else:
                return [idx.name for idx in indexes]
        else:
            # Pinecone v2.x
            return pinecone.list_indexes()
    except Exception as e:
        print(f"Error listing Pinecone indexes: {e}")
        return []

def delete_pinecone_index(index_name: str):
    """Delete Pinecone index with version compatibility."""
    try:
        if PINECONE_V3:
            # Pinecone v3.x
            pc.delete_index(index_name)
        else:
            # Pinecone v2.x
            pinecone.delete_index(index_name)
        print(f"Deleted Pinecone index: {index_name}")
    except Exception as e:
        print(f"Error deleting Pinecone index '{index_name}': {e}")
        raise

# FIXED: Retriever function with proper error handling
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""
    try:
        print(f"Initializing retriever for index: {INDEX_NAME}")
        
        # Check if index exists
        existing_indexes = list_pinecone_indexes()
        print(f"Existing indexes: {existing_indexes}")
        
        if INDEX_NAME not in existing_indexes:
            print(f"Creating new Pinecone index: {INDEX_NAME}...")
            create_pinecone_index(INDEX_NAME, dimension=1024)
            
            # Wait a moment for index to be ready
            import time
            time.sleep(10)
        
        # Get the index
        index = get_pinecone_index(INDEX_NAME)
        print(f"Successfully got Pinecone index: {INDEX_NAME}")
        
        # Create vector store using the legacy approach
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        print("Successfully created vectorstore")
        return vectorstore.as_retriever()
        
    except Exception as e:
        error_msg = f"RAG_ERROR::{e}"
        print(error_msg)
        raise Exception(error_msg)

# FIXED: Function to add documents with proper error handling
def add_document_to_vectorstore(text_content: str):
    """Adds a single text document to the Pinecone vector store."""
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    try:
        print("Starting document addition to vectorstore...")
        
        # Ensure index exists
        get_retriever()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            add_start_index=True,
        )
        
        documents = text_splitter.create_documents([text_content])
        print(f"Splitting document into {len(documents)} chunks for indexing...")
        
        # Get the index
        index = get_pinecone_index(INDEX_NAME)
        
        # Create vector store
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        print("Adding documents to vectorstore...")
        vectorstore.add_documents(documents)
        print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
        
        return len(documents)
        
    except Exception as e:
        error_msg = f"Error adding document to vectorstore: {e}"
        print(error_msg)
        raise Exception(f"Failed to add document: {e}")

# FIXED: Add a health check function
def check_vectorstore_health():
    """Check if the vectorstore is working properly."""
    try:
        print("Checking vectorstore health...")
        
        # Check if we can list indexes
        indexes = list_pinecone_indexes()
        print(f"Available indexes: {indexes}")
        
        # Check if our index exists
        if INDEX_NAME not in indexes:
            return False, f"Index '{INDEX_NAME}' does not exist"
        
        # Try to get the index
        index = get_pinecone_index(INDEX_NAME)
        
        # Try to get index stats
        if PINECONE_V3:
            stats = index.describe_index_stats()
        else:
            stats = index.describe_index_stats()
            
        print(f"Index stats: {stats}")
        
        # Try to create vectorstore
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Try a simple retrieval (this will work even if index is empty)
        retriever = vectorstore.as_retriever()
        
        return True, "Vectorstore is healthy"
        
    except Exception as e:
        error_msg = f"Vectorstore health check failed: {e}"
        print(error_msg)
        return False, error_msg

# Initialize vectorstore on module load
def initialize_vectorstore():
    """Initialize vectorstore on module load."""
    try:
        print("Initializing vectorstore...")
        get_retriever()
        print("Vectorstore initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize vectorstore: {e}")

# Call initialization
# initialize_vectorstore()  # Commented out to avoid issues during imports

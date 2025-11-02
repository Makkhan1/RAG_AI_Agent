# rag_agent_app/backend/vectorstore.py

import os
import requests
import json
from typing import List

from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- CHANGE 1: Import the base Embeddings class to create our custom one ---
from langchain_core.embeddings import Embeddings

# --- CHANGE 2: Import the JINA_API_KEY from config ---
from .config import PINECONE_API_KEY, JINA_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index name
INDEX_NAME = "rag-index"

# --- CHANGE 3: Create a custom LangChain Embeddings class for Jina AI ---
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
            print(f"Received response text: {response.text}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents for storage."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query for retrieval."""
        result = self._embed([text])
        return result[0] if result else []

# --- CHANGE 4: Initialize our new custom Jina embeddings class ---
embeddings = JinaApiEmbeddings(api_key=JINA_API_KEY)


# --- Retriever (Updated function with FIX) ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            # --- CHANGE 5: CRITICAL - Update the dimension for Jina's model ---
            dimension=1024, # Your chosen dimension for jina-embeddings-v3
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")
    
    # FIX: Get the actual index object and add text_key
    index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index=index, 
        embedding=embeddings,
        text_key="text"  # ADD THIS: Required parameter for legacy Pinecone
    )
    return vectorstore.as_retriever()

# --- Function to add documents (Updated with FIX) ---
def add_document_to_vectorstore(text_content: str):
    """Adds a single text document to the Pinecone vector store."""
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    get_retriever() # Ensures index exists

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, # Your updated chunk size
        chunk_overlap=150, # Your updated chunk overlap
        add_start_index=True,
    )
    
    documents = text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for indexing...")
    
    # FIX: Get the actual index object and add text_key
    index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index=index, 
        embedding=embeddings,
        text_key="text"  # ADD THIS: Required parameter for legacy Pinecone
    )
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")

import os
import requests
import json
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# ABSOLUTE IMPORTS
from config import PINECONE_API_KEY, JINA_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index name
INDEX_NAME = "rag-index"

class JinaApiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "jina-embeddings-v3", task: str = "text-matching"):
        if not api_key:
            raise ValueError("Jina API key not found.")
        self.api_key = api_key
        self.model_name = model_name
        self.task = task
        self.api_url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def _embed(self, texts: List[str]) -> List[List[float]]:
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
            response.raise_for_status()
            
            result = response.json()
            if "data" not in result or not result["data"]:
                raise ValueError(f"Jina API returned no data.")

            sorted_embeddings = sorted(result["data"], key=lambda e: e["index"])
            return [item["embedding"] for item in sorted_embeddings]

        except Exception as e:
            print(f"Error calling Jina API: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._embed([text])
        return result[0] if result else []

embeddings = JinaApiEmbeddings(api_key=JINA_API_KEY)

def get_retriever(namespace: Optional[str] = None):
    """
    Returns a retriever. 
    Note: agent.py now instantiates its own vectorstore with config-based namespaces,
    but this helper is kept for backward compatibility or testing.
    """
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings,
        namespace=namespace
    )
    return vectorstore.as_retriever()

def add_document_to_vectorstore(text_content: str, namespace: str):
    """
    Adds a document to the specified Namespace (Session ID).
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")
    
    # Ensure index exists (lazy check)
    if INDEX_NAME not in pc.list_indexes().names():
        get_retriever() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        add_start_index=True,
    )
    
    documents = text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for namespace '{namespace}'...")
    
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings,
        namespace=namespace # <--- CRITICAL: Isolates data per session
    )
    vectorstore.add_documents(documents)
    print(f"Successfully added chunks to namespace: {namespace}")

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT","us-east-1") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index") 

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Jina
JINA_API_KEY = os.getenv("JINA_API_KEY")

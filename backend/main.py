import io
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, AIMessage

# ABSOLUTE IMPORTS
from agent import rag_agent
from vectorstore import add_document_to_vectorstore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent API", version="1.0.0")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    query: str
    enable_web_search: bool = True

class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent]

# --- Helper: PDF Extraction ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise HTTPException(status_code=400, detail="Invalid PDF file")

# --- Endpoints ---

@app.get("/health")
async def root():
    return {"status": "ok", "message": "RAG Agent Backend is running"}

@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG Agent Backend is running"}

@app.post("/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...) # <--- NEW: Require session_id for isolation
):
    """
    Accepts a PDF file and session_id.
    Extracts text and adds it to the Pinecone namespace matching the session_id.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    logger.info(f"Receiving file upload: {file.filename} for session: {session_id}")
    
    try:
        content = await file.read()
        text_content = extract_text_from_pdf(content)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")
            
        # Add to vector store with Namespace isolation
        add_document_to_vectorstore(text_content, namespace=session_id)
        
        return {
            "filename": file.filename, 
            "status": "success", 
            "message": "Document added to session knowledge base."
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Runs the LangGraph agent and returns the final answer + execution trace.
    """
    logger.info(f"Chat request for session {request.session_id}: {request.query}")
    
    inputs = {"messages": [HumanMessage(content=request.query)]}
    
    # Config passes session_id as thread_id (used for MemorySaver AND Namespace lookup)
    config = {
        "configurable": {
            "thread_id": request.session_id,
            "web_search_enabled": request.enable_web_search
        }
    }
    
    trace_events = []
    step_count = 1
    final_response = ""
    
    try:
        for chunk in rag_agent.stream(inputs, config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                
                description = ""
                details = {}
                
                if node_name == "router":
                    route = node_output.get("route")
                    description = f"Routed query to: {route.upper()}"
                    details["route"] = route
                    if "router_override_reason" in node_output:
                        details["router_override_reason"] = node_output["router_override_reason"]
                        
                elif node_name == "rag_lookup":
                    is_sufficient = node_output.get("rag_verdict_is_sufficient", False)
                    doc_preview = node_output.get("rag", "")[:50] + "..." if node_output.get("rag") else "No docs"
                    verdict_str = "Sufficient" if is_sufficient else "Insufficient"
                    
                    description = f"Checked Knowledge Base. Verdict: {verdict_str}"
                    details["sufficiency_verdict"] = verdict_str
                    details["retrieved_content_summary"] = doc_preview
                    
                elif node_name == "web_search":
                    web_res = node_output.get("web", "")
                    preview = web_res[:50] + "..." if web_res else "No results"
                    description = "Performed Web Search via Tavily."
                    details["retrieved_content_summary"] = preview
                    
                elif node_name == "answer":
                    description = "Generated final response using available context."
                
                trace_events.append(TraceEvent(
                    step=step_count,
                    node_name=node_name,
                    description=description,
                    details=details
                ))
                step_count += 1
        
        final_state = rag_agent.get_state(config)
        messages = final_state.values.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            final_response = messages[-1].content
        else:
            final_response = "Error: No response generated."

        return ChatResponse(response=final_response, trace_events=trace_events)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

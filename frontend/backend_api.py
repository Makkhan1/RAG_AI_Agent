# rag_agent_app/frontend/backend_api.py

import requests
import json
import time
import streamlit as st

def check_backend_health(fastapi_base_url: str) -> bool:
    """
    Checks if the backend server is healthy and responding.
    """
    try:
        response = requests.get(f"{fastapi_base_url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_vectorstore_health(fastapi_base_url: str) -> tuple:
    """
    Checks if the vectorstore is healthy.
    """
    try:
        response = requests.get(f"{fastapi_base_url}/vectorstore-health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy", data.get("message", "Unknown status")
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def upload_document_to_backend(fastapi_base_url: str, uploaded_file):
    """
    Sends a PDF document to the FastAPI backend for upload and indexing.
    """
    # Validate file size (50MB limit)
    file_size = len(uploaded_file.getvalue())
    max_size = 50 * 1024 * 1024  # 50MB
    
    if file_size > max_size:
        raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f}MB). Please use a file smaller than 50MB.")
    
    # Prepare the file for a multipart/form-data request
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    try:
        response = requests.post(
            f"{fastapi_base_url}/upload-document/", 
            files=files,
            timeout=300  # 5 minutes
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout("Upload timed out after 5 minutes. Please try with a smaller file.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 408:
            raise requests.exceptions.Timeout("Upload timeout - please try again")
        elif e.response.status_code == 413:
            raise ValueError("File too large - please use a smaller file")
        elif e.response.status_code == 502:
            raise requests.exceptions.HTTPError("Server error - the backend may be starting up. Please wait and try again.")
        else:
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", error_detail)
            except:
                pass
            raise requests.exceptions.HTTPError(f"Upload failed: {error_detail}")

def chat_with_backend_agent(fastapi_base_url: str, session_id: str, query: str, enable_web_search: bool):
    """
    Sends a chat query to the FastAPI backend's agent.
    """
    payload = {
        "session_id": session_id,
        "query": query,
        "enable_web_search": enable_web_search
    }
    
    # Retry logic for transient failures
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{fastapi_base_url}/chat/", 
                json=payload,
                timeout=120,  # 2 minutes
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            agent_response = data.get("response", "Sorry, I couldn't get a response from the agent.")
            trace_events = data.get("trace_events", [])
            
            return agent_response, trace_events
            
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                raise requests.exceptions.Timeout("Chat request timed out. Please try again.")
            else:
                time.sleep(2)
                continue
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 408:
                raise requests.exceptions.Timeout("Chat timeout - please try again")
            elif e.response.status_code == 502:
                if attempt == max_retries:
                    raise requests.exceptions.HTTPError("Server error - the backend may be experiencing issues.")
                else:
                    time.sleep(5)
                    continue
            elif e.response.status_code >= 500:
                if attempt == max_retries:
                    error_detail = "Internal server error"
                    try:
                        error_data = e.response.json()
                        error_detail = error_data.get("detail", error_detail)
                    except:
                        pass
                    raise requests.exceptions.HTTPError(f"Server error: {error_detail}")
                else:
                    time.sleep(2)
                    continue
            else:
                error_detail = "Unknown error"
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise requests.exceptions.HTTPError(f"Chat failed: {error_detail}")
                
        except requests.exceptions.ConnectionError:
            if attempt == max_retries:
                raise requests.exceptions.ConnectionError("Cannot connect to backend server.")
            else:
                time.sleep(3)
                continue

def upload_document_with_progress(fastapi_base_url: str, uploaded_file):
    """
    Upload document with progress tracking.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Checking backend status...")
        progress_bar.progress(10)
        
        if not check_backend_health(fastapi_base_url):
            raise Exception("Backend server is not responding")
        
        status_text.text("📤 Uploading document...")
        progress_bar.progress(25)
        
        time.sleep(1)
        progress_bar.progress(50)
        status_text.text("📝 Processing document...")
        
        result = upload_document_to_backend(fastapi_base_url, uploaded_file)
        
        progress_bar.progress(75)
        status_text.text("🔍 Adding to knowledge base...")
        time.sleep(1)
        
        progress_bar.progress(100)
        status_text.text("✅ Upload complete!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return result
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        raise e

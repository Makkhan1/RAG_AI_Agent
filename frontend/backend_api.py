# rag_agent_app/frontend/backend_api.py

import requests
import json
import time
import streamlit as st

# Request timeout settings
UPLOAD_TIMEOUT = 300  # 5 minutes for document upload
CHAT_TIMEOUT = 120    # 2 minutes for chat
HEALTH_TIMEOUT = 10   # 10 seconds for health check

def check_backend_health(fastapi_base_url: str) -> bool:
    """
    Checks if the backend server is healthy and responding.
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        
    Returns:
        bool: True if backend is healthy, False otherwise.
    """
    try:
        response = requests.get(f"{fastapi_base_url}/health", timeout=HEALTH_TIMEOUT)
        return response.status_code == 200
    except:
        return False

def check_vectorstore_health(fastapi_base_url: str) -> tuple:
    """
    Checks if the vectorstore is healthy.
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        
    Returns:
        tuple: (is_healthy: bool, message: str)
    """
    try:
        response = requests.get(f"{fastapi_base_url}/vectorstore-health", timeout=HEALTH_TIMEOUT)
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
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): The file object from Streamlit's file_uploader.
        
    Returns:
        dict: The JSON response from the backend on success.
        
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        json.JSONDecodeError: If the response is not valid JSON.
    """
    
    # Validate file size (50MB limit)
    file_size = len(uploaded_file.getvalue())
    max_size = 50 * 1024 * 1024  # 50MB
    
    if file_size > max_size:
        raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f}MB). Please use a file smaller than 50MB.")
    
    # Prepare the file for a multipart/form-data request
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    # Make a POST request to the backend's upload endpoint with timeout
    try:
        response = requests.post(
            f"{fastapi_base_url}/upload-document/", 
            files=files,
            timeout=UPLOAD_TIMEOUT
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(
            "Upload timed out after 5 minutes. Please try with a smaller file or try again later."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 408:
            raise requests.exceptions.Timeout("Upload timeout - please try again with a smaller file")
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
    Sends a chat query to the FastAPI backend's agent with retry logic.
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        session_id (str): Unique ID for the current chat session.
        query (str): The user's chat message.
        enable_web_search (bool): Flag indicating if web search is enabled.
        
    Returns:
        tuple: (agent_response_text: str, trace_events: list)
        
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        json.JSONDecodeError: If the response is not valid JSON.
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
                timeout=CHAT_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            
            data = response.json()
            agent_response = data.get("response", "Sorry, I couldn't get a response from the agent.")
            trace_events = data.get("trace_events", [])
            
            return agent_response, trace_events
            
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                raise requests.exceptions.Timeout(
                    "Chat request timed out after 2 minutes. Please try a shorter question or try again later."
                )
            else:
                time.sleep(2)  # Wait 2 seconds before retry
                continue
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 408:
                raise requests.exceptions.Timeout("Chat timeout - please try again")
            elif e.response.status_code == 502:
                if attempt == max_retries:
                    raise requests.exceptions.HTTPError("Server error - the backend may be experiencing issues. Please try again later.")
                else:
                    time.sleep(5)  # Wait longer for 502 errors
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
                    time.sleep(2)  # Wait before retry for server errors
                    continue
            else:
                # Client errors (4xx) - don't retry
                error_detail = "Unknown error"
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise requests.exceptions.HTTPError(f"Chat failed: {error_detail}")
                
        except requests.exceptions.ConnectionError:
            if attempt == max_retries:
                raise requests.exceptions.ConnectionError(
                    "Cannot connect to backend server. Please check if the server is running."
                )
            else:
                time.sleep(3)  # Wait longer for connection issues
                continue

def upload_document_with_progress(fastapi_base_url: str, uploaded_file):
    """
    Upload document with progress tracking using Streamlit.
    
    Args:
        fastapi_base_url (str): The base URL of the FastAPI backend.
        uploaded_file: The uploaded file object.
        
    Returns:
        dict: Upload response data.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Checking backend status...")
        progress_bar.progress(10)
        
        # Check backend health first
        if not check_backend_health(fastapi_base_url):
            raise Exception("Backend server is not responding")
        
        status_text.text("📤 Uploading document...")
        progress_bar.progress(25)
        
        # Simulate upload progress (since requests doesn't provide real progress)
        time.sleep(1)
        progress_bar.progress(50)
        status_text.text("📝 Processing document...")
        
        # Actual upload
        result = upload_document_to_backend(fastapi_base_url, uploaded_file)
        
        progress_bar.progress(75)
        status_text.text("🔍 Adding to knowledge base...")
        time.sleep(1)
        
        progress_bar.progress(100)
        status_text.text("✅ Upload complete!")
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return result
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        raise e

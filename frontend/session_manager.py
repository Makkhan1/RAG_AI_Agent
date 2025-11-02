# rag_agent_app/frontend/session_manager.py

import streamlit as st
import uuid
import time

def init_session_state():
    """
    Initializes necessary variables in Streamlit's session state.
    This ensures data persists across reruns of the Streamlit app.
    """
    # Initialize chat message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize a unique session ID for LangGraph checkpointing and conversation tracking
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        # Add an initial greeting from the assistant for a fresh conversation
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

    # Initialize the web search enabled flag, defaulting to True
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True
    
    # Initialize upload tracking
    if "upload_count" not in st.session_state:
        st.session_state.upload_count = 0
    
    # Initialize error tracking
    if "consecutive_errors" not in st.session_state:
        st.session_state.consecutive_errors = 0
    
    # Initialize successful operations
    if "successful_operations" not in st.session_state:
        st.session_state.successful_operations = 0

def track_error():
    """Track an error occurrence."""
    st.session_state.consecutive_errors += 1

def reset_error_count():
    """Reset error count after successful operation."""
    st.session_state.consecutive_errors = 0
    st.session_state.successful_operations += 1

def track_upload():
    """Track a successful upload."""
    st.session_state.upload_count += 1
    reset_error_count()

def should_show_help():
    """Determine if help message should be shown based on errors."""
    return st.session_state.consecutive_errors >= 3

def get_session_stats():
    """Get session statistics for display."""
    return {
        "session_id": st.session_state.session_id,
        "message_count": len(st.session_state.messages),
        "web_search_enabled": st.session_state.web_search_enabled,
        "upload_count": st.session_state.upload_count,
        "consecutive_errors": st.session_state.consecutive_errors,
        "successful_operations": st.session_state.successful_operations
    }

def reset_session():
    """Reset the current session and start fresh."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    st.session_state.consecutive_errors = 0
    st.session_state.upload_count = 0
    st.session_state.successful_operations = 0

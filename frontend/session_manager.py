# rag_agent_app/frontend/session_manager.py

import streamlit as st
import uuid

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
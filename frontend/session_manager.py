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

    # --- FIX: Initialize new variables required by the updated app.py ---

    # Flag to control the chat input's disabled state while the agent is thinking.
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Stores the trace from the last backend call. Initialize as an empty list.
    # This is the direct fix for the AttributeError you encountered.
    if "trace_events" not in st.session_state:
        st.session_state.trace_events = []
        
    # Used to handle clicks from the example question buttons.
    if "prompt_from_button" not in st.session_state:
        st.session_state.prompt_from_button = None

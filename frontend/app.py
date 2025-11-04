# rag_agent_app/frontend/app.py

import streamlit as st
import requests
import json
from config import FRONTEND_CONFIG
from session_manager import init_session_state
from ui_components import (
    display_header, 
    render_document_upload_section, 
    render_agent_settings_section, 
    display_chat_history, 
    display_trace_events,
    render_example_questions
)
from backend_api import chat_with_backend_agent

def main():
    """Main function to run the Streamlit application."""
    
    init_session_state()
    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    # --- 1. RENDER STATIC UI COMPONENTS ---
    display_header()
    render_document_upload_section(fastapi_base_url)
    render_agent_settings_section()

    # --- 2. RENDER THE "CHAT ARENA" ---
    st.subheader("💬 Conversation")
    with st.container(border=True):
        display_chat_history()
        # Show example questions only in a new chat
        if len(st.session_state.messages) == 1:
            render_example_questions()

    # --- 3. HANDLE USER INPUT ---
    # This block captures input from either the text input or an example button click.
    if prompt := st.session_state.pop("prompt_from_button", None) or st.chat_input(
        "Ask about your documents or a general question...", 
        disabled=st.session_state.get("processing", False),
        key="main_chat_input"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Set processing flag and rerun to immediately disable input and show user message
        st.session_state.processing = True
        st.rerun()

    # --- 4. HANDLE BACKEND CALL ---
    # This block runs ONLY if we are in a "processing" state and the last message was from the user.
    if st.session_state.get("processing") and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                try:
                    user_prompt = st.session_state.messages[-1]["content"]
                    
                    # Call the backend API using the signature from your original file
                    agent_response, trace_events = chat_with_backend_agent(
                        fastapi_base_url,
                        st.session_state.session_id,
                        user_prompt,
                        st.session_state.web_search_enabled
                    )
                    
                    st.markdown(agent_response)
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})
                    st.session_state.trace_events = trace_events

                # Preserve your excellent, detailed error handling
                except requests.exceptions.ConnectionError:
                    error_msg = "Error: Could not connect to the backend. Please ensure it's running."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except requests.exceptions.RequestException as e:
                    error_msg = f"Error with the request: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except json.JSONDecodeError:
                    error_msg = "Error: Received an invalid response from the backend."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Reset processing flag and rerun to re-enable the input field
        st.session_state.processing = False
        st.rerun()

    # --- 5. DISPLAY THE TRACE ---
    # This is now outside the chat message bubble for a cleaner look.
    if st.session_state.trace_events:
        display_trace_events(st.session_state.trace_events)

if __name__ == "__main__":
    main()

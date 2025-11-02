# rag_agent_app/frontend/app.py

import streamlit as st
import requests
from config import FRONTEND_CONFIG
from session_manager import init_session_state
from ui_components import (
    display_header, 
    render_document_upload_section, 
    render_agent_settings_section, 
    display_chat_history, 
    display_trace_events
)
from backend_api import chat_with_backend_agent

def main():
    """Main function to run the Streamlit application."""
    
    # Initialize session state variables
    init_session_state()

    # Get FastAPI base URL from config
    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    # Render UI sections
    display_header()
    render_document_upload_section(fastapi_base_url)
    render_agent_settings_section()

    st.header("Chat with the Agent")
    display_chat_history()

    # User input field
    if prompt := st.chat_input("Your message"):
        # Add user's message to chat history and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant's response and trace
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call the backend API for chat
                    agent_response, trace_events = chat_with_backend_agent(
                        fastapi_base_url,
                        st.session_state.session_id,
                        prompt,
                        st.session_state.web_search_enabled
                    )
                    
                    # Display the agent's final response
                    st.markdown(agent_response)
                    # Add the agent's response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})

                    # Display the workflow trace
                    display_trace_events(trace_events)
                    
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure it's running.")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Could not connect to the backend."})
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred with the request: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                except json.JSONDecodeError:
                    st.error("Received an invalid response from the backend.")
                    st.session_state.messages.append({"role": "assistant", "content": "Error: Invalid response from backend."})
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Unexpected Error: {e}"})

if __name__ == "__main__":
    main()

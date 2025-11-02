# rag_agent_app/frontend/app.py

import streamlit as st
import requests
import json
import time
from config import FRONTEND_CONFIG
from session_manager import init_session_state
from ui_components import (
    display_header, 
    display_backend_status,
    render_document_upload_section, 
    render_agent_settings_section, 
    display_chat_history, 
    display_trace_events
)
from backend_api import chat_with_backend_agent, check_backend_health

def main():
    """Main function to run the Streamlit application."""
    
    # Initialize session state variables
    init_session_state()

    # Get FastAPI base URL from config
    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    # Render UI sections
    display_header()
    
    # Display backend status at the top
    backend_healthy = display_backend_status(fastapi_base_url)
    
    render_document_upload_section(fastapi_base_url)
    render_agent_settings_section()

    st.header("💬 Chat with the Agent")
    display_chat_history()

    # Disable chat if backend is offline
    chat_disabled = not backend_healthy

    if chat_disabled:
        st.warning("🚫 Chat is disabled because the backend is offline. Please wait for it to come online.")
        if st.button("🔄 Check Backend Status"):
            st.rerun()

    # User input field
    if prompt := st.chat_input("Your message", disabled=chat_disabled):
        # Add user's message to chat history and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant's response and trace
        with st.chat_message("assistant"):
            # Create placeholders for dynamic updates
            message_placeholder = st.empty()
            trace_placeholder = st.empty()
            
            with message_placeholder:
                with st.spinner("🤖 AI is thinking... This may take up to 2 minutes."):
                    try:
                        # Call the backend API for chat
                        agent_response, trace_events = chat_with_backend_agent(
                            fastapi_base_url,
                            st.session_state.session_id,
                            prompt,
                            st.session_state.web_search_enabled
                        )
                        
                        # Clear spinner and display response
                        message_placeholder.markdown(agent_response)
                        
                        # Add the agent's response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": agent_response})

                        # Display the workflow trace
                        with trace_placeholder:
                            display_trace_events(trace_events)
                        
                    except requests.exceptions.Timeout:
                        error_msg = "⏱️ Request timed out. The backend may be processing a complex query. Please try again."
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        
                    except requests.exceptions.ConnectionError:
                        error_msg = "🔌 Could not connect to the backend server. Please ensure it's running and try again."
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        # Update backend status
                        if st.button("🔄 Retry Connection"):
                            st.rerun()
                        
                    except requests.exceptions.HTTPError as e:
                        if "502" in str(e):
                            error_msg = "🔧 Backend server error. The server may be starting up. Please wait and try again."
                        elif "408" in str(e):
                            error_msg = "⏱️ Request timeout. Please try again with a shorter question."
                        elif "500" in str(e):
                            error_msg = "🔧 Internal server error. Please try again or contact support."
                        else:
                            error_msg = f"🚨 HTTP Error: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        
                    except requests.exceptions.RequestException as e:
                        error_msg = f"🌐 Network error occurred: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        
                    except json.JSONDecodeError:
                        error_msg = "📋 Received invalid response from backend. Please try again."
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        
                    except Exception as e:
                        error_msg = f"💥 Unexpected error: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})

    # Add sidebar with additional controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
            st.rerun()
            
        if st.button("🔄 Refresh Status"):
            st.rerun()
            
        st.header("ℹ️ Session Info")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        st.write(f"**Web Search:** {'Enabled' if st.session_state.web_search_enabled else 'Disabled'}")
        
        # Backend info
        st.header("🖥️ Backend Info")
        st.write(f"**URL:** {fastapi_base_url}")
        st.write(f"**Status:** {'🟢 Online' if backend_healthy else '🔴 Offline'}")

if __name__ == "__main__":
    main()

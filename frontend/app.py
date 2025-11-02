# rag_agent_app/frontend/app.py

import streamlit as st
import requests
import json
from config import FRONTEND_CONFIG
from session_manager import init_session_state, track_error, reset_error_count
from ui_components import (
    display_header, 
    display_backend_status,
    render_document_upload_section, 
    render_agent_settings_section, 
    display_chat_history, 
    display_trace_events
)
from backend_api import chat_with_backend_agent

def main():
    """Main function to run the Streamlit application."""
    
    init_session_state()
    fastapi_base_url = FRONTEND_CONFIG["FASTAPI_BASE_URL"]

    display_header()
    backend_healthy = display_backend_status(fastapi_base_url)
    render_document_upload_section(fastapi_base_url)
    render_agent_settings_section()

    st.header("💬 Chat with the Agent")
    display_chat_history()

    chat_disabled = not backend_healthy

    if chat_disabled:
        st.warning("🚫 Chat is disabled because the backend is offline.")
        if st.button("🔄 Check Backend Status"):
            st.rerun()

    if prompt := st.chat_input("Your message", disabled=chat_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            trace_placeholder = st.empty()
            
            with message_placeholder:
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        agent_response, trace_events = chat_with_backend_agent(
                            fastapi_base_url,
                            st.session_state.session_id,
                            prompt,
                            st.session_state.web_search_enabled
                        )
                        
                        message_placeholder.markdown(agent_response)
                        st.session_state.messages.append({"role": "assistant", "content": agent_response})
                        reset_error_count()

                        with trace_placeholder:
                            display_trace_events(trace_events)
                        
                    except requests.exceptions.Timeout:
                        error_msg = "⏱️ Request timed out. Please try again."
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        track_error()
                        
                    except requests.exceptions.ConnectionError:
                        error_msg = "🔌 Could not connect to the backend server."
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        track_error()
                        
                    except requests.exceptions.HTTPError as e:
                        if "502" in str(e):
                            error_msg = "🔧 Backend server error. Please wait and try again."
                        elif "408" in str(e):
                            error_msg = "⏱️ Request timeout. Please try again."
                        else:
                            error_msg = f"🚨 HTTP Error: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        track_error()
                        
                    except Exception as e:
                        error_msg = f"💥 Unexpected error: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        track_error()

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

if __name__ == "__main__":
    main()

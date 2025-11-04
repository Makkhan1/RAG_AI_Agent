# rag_agent_app/frontend/ui_components.py

import streamlit as st
from backend_api import upload_document_to_backend

def display_header():
    """Renders the main title and introductory markdown with a premium, centered feel."""
    st.set_page_config(
        page_title="RAG AI Chatbot", 
        layout="wide", 
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("<h1 style='text-align: center;'>🤖 RAG AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your intelligent assistant for querying documents and the web.</p>", unsafe_allow_html=True)
    
    st.divider()

def render_document_upload_section(fastapi_base_url: str):
    """Renders a premium-looking UI for uploading documents."""
    st.subheader("📚 Manage Knowledge Base")
    
    with st.container(border=True):
        with st.expander("Upload New Document (PDF Only)", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a PDF file to add to the agent's knowledge.", 
                    type="pdf", 
                    key="pdf_uploader",
                    label_visibility="collapsed"
                )
            
            with col2:
                if st.button("Process & Add to Knowledge", key="upload_pdf_button", use_container_width=True):
                    if uploaded_file is not None:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                # Note: Your original backend_api.py might need adjustment if it returns different data
                                upload_data = upload_document_to_backend(fastapi_base_url, uploaded_file)
                                st.success(f"Success! Added '{upload_data.get('filename', uploaded_file.name)}' to knowledge base.")
                            except Exception as e:
                                st.error(f"Upload failed: {e}")
                    else:
                        st.warning("Please select a PDF file first.")
    st.divider()

def render_agent_settings_section():
    """Renders a premium-looking agent settings section with a toggle switch."""
    st.subheader("⚙️ Agent Settings")
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Agent's Reach**")
            st.caption("Control whether the agent can access the live web for answers or is restricted to only the uploaded documents.")
        
        with col2:
            is_enabled = st.toggle(
                "Web Search", 
                value=st.session_state.get("web_search_enabled", True),
                key="web_search_toggle",
                help="ON: Agent can use Google. OFF: Agent is restricted to documents."
            )
            st.session_state.web_search_enabled = is_enabled

        if st.session_state.web_search_enabled:
            st.success("🌐 Web search is **ACTIVE**. The agent can access real-time information.")
        else:
            st.warning("📚 Web search is **DISABLED**. The agent will only use the knowledge base.")
            
    st.divider()

def display_chat_history():
    """Displays all messages currently in the session state chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_trace_events(trace_events: list):
    """Renders the detailed agent workflow trace in a more structured and readable format."""
    if trace_events:
        with st.expander("🔬 View Agent's Thought Process"):
            for event in trace_events:
                with st.container(border=True):
                    icon_map = {'router': "🧭", 'rag_lookup': "📚", 'web_search': "🌐", 'answer': "💡", '__end__': "✅"}
                    icon = icon_map.get(event['node_name'], "⚙️")
                    
                    st.markdown(f"**{icon} Step {event['step']}: `{event['node_name']}`**")
                    st.caption(event['description'])
                    
                    details = event.get('details', {})
                    
                    if 'sufficiency_verdict' in details:
                        verdict = details['sufficiency_verdict']
                        if verdict == "Sufficient":
                            st.success(f"**Verdict:** {verdict} - Found relevant info in the knowledge base.")
                        else:
                            st.warning(f"**Verdict:** {verdict} - No sufficient info found. Will try another tool.")
                        if 'retrieved_content_summary' in details:
                            st.info(f"**Retrieved:** `{details['retrieved_content_summary']}`")
                    elif 'retrieved_content_summary' in details:
                        st.info(f"**Retrieved:** `{details['retrieved_content_summary']}`")
                    elif 'router_override_reason' in details:
                        st.info(f"**Router Override:** {details['router_override_reason']}")
                        st.json({"Initial Decision": details['initial_decision'], "Final Decision": details['final_decision']})
                    elif details and not any(k in details for k in ['sufficiency_verdict', 'retrieved_content_summary', 'router_override_reason']):
                        st.json(details)

def render_example_questions():
    """Displays clickable example questions to guide the user."""
    st.markdown("##### Or try one of these examples:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What is the main topic of the uploaded document?", use_container_width=True):
            st.session_state.prompt_from_button = "What is the main topic of the uploaded document?"
            st.rerun()
    with col2:
        if st.button("Summarize the document in three bullet points.", use_container_width=True):
            st.session_state.prompt_from_button = "Summarize the document in three bullet points."
            st.rerun()
    with col3:
        if st.button("What are the latest trends in AI?", use_container_width=True):
            st.session_state.prompt_from_button = "What are the latest trends in AI?"
            st.rerun()

# rag_agent_app/frontend/ui_components.py

import streamlit as st
import time
from backend_api import upload_document_with_progress, check_backend_health, check_vectorstore_health
from session_manager import track_error, reset_error_count, track_upload, should_show_help, get_session_stats
from config import FRONTEND_CONFIG

def display_header():
    """Renders the main title and introductory markdown."""
    st.set_page_config(
        page_title="AI Agent Chatbot", 
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Agent Chatbot")
    st.markdown("""
    **Welcome to your intelligent AI assistant!** 
    
    I can help you by:
    - 📚 Answering questions about uploaded documents (RAG)
    - 🌐 Searching the web for current information
    - 💡 Providing detailed explanations with source tracing
    """)
    
    # Show help if user is having issues
    if should_show_help():
        st.warning("""
        **Having trouble?** Here are some tips:
        - Make sure your PDF files are smaller than 50MB
        - Wait for uploads to complete before asking questions
        - Try shorter, more specific questions
        - Check that the backend server is online (green status below)
        """)
    
    st.markdown("---")

def display_backend_status(fastapi_base_url: str):
    """Display backend and vectorstore status."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ Backend Status")
        backend_healthy = check_backend_health(fastapi_base_url)
        
        if backend_healthy:
            st.success("🟢 Backend Online")
        else:
            st.error("🔴 Backend Offline")
            st.info("💡 If using Render, the backend may take 1-2 minutes to wake up from sleep.")
    
    with col2:
        st.subheader("🗃️ Vectorstore Status")
        if backend_healthy:
            vectorstore_healthy, vectorstore_msg = check_vectorstore_health(fastapi_base_url)
            
            if vectorstore_healthy:
                st.success(f"🟢 Vectorstore Healthy")
                st.caption(vectorstore_msg)
            else:
                st.warning(f"🟡 Vectorstore Issues")
                st.caption(vectorstore_msg)
        else:
            st.info("❓ Cannot check - backend offline")
    
    return backend_healthy

def render_document_upload_section(fastapi_base_url: str):
    """
    Renders the UI for uploading PDF documents to the knowledge base.
    Handles file upload and API call to the backend with improved UX.
    """
    st.header("📤 Upload Document to Knowledge Base")
    
    # Check backend status first
    backend_healthy = check_backend_health(fastapi_base_url)
    
    # Show upload statistics
    stats = get_session_stats()
    if stats["upload_count"] > 0:
        st.info(f"📊 Documents uploaded this session: {stats['upload_count']}")
    
    with st.expander("Upload New Document (PDF Only)", expanded=stats["upload_count"] == 0):
        if not backend_healthy:
            st.error("❌ Cannot upload: Backend server is offline")
            if st.button("🔄 Check Backend Status", key="check_backend_upload"):
                st.rerun()
            return
        
        # File uploader with validation
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf", 
            key="pdf_uploader",
            help="Supported: PDF files up to 50MB"
        )
        
        # Show file info if file is selected
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File:** {uploaded_file.name}")
                st.write(f"**Size:** {file_size_mb:.1f} MB")
            
            with col2:
                if file_size_mb > 50:
                    st.error("❌ File too large! Please use a file smaller than 50MB.")
                    upload_disabled = True
                else:
                    st.success("✅ File size acceptable")
                    upload_disabled = False
        else:
            upload_disabled = True
        
        # Upload button with validation
        if st.button("📤 Upload PDF", key="upload_pdf_button", disabled=upload_disabled):
            if uploaded_file is not None:
                try:
                    # Upload with progress
                    upload_data = upload_document_with_progress(fastapi_base_url, uploaded_file)
                    
                    # Success message
                    st.success(f"""
                    ✅ **Upload Successful!**
                    - File: {upload_data.get('filename')}
                    - Processed: {upload_data.get('processed_chunks')} chunks
                    - Status: Ready for questions!
                    """)
                    
                    # Track successful upload
                    track_upload()
                    reset_error_count()
                    
                    # Auto-clear file uploader after successful upload
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Upload failed: {error_msg}")
                    track_error()
                    
                    # Provide specific help based on error type
                    if "timeout" in error_msg.lower():
                        st.info("💡 **Tip:** Large files may take several minutes to process. Try a smaller file or wait and retry.")
                    elif "connection" in error_msg.lower() or "502" in error_msg:
                        st.info("💡 **Tip:** The backend server may be starting up. Please wait 1-2 minutes and try again.")
                    elif "too large" in error_msg.lower():
                        st.info("💡 **Tip:** Please use a PDF file smaller than 50MB.")
                    elif "backend" in error_msg.lower():
                        st.info("💡 **Tip:** Wait for the backend to come online (check status above) and try again.")
            else:
                st.warning("⚠️ Please select a PDF file before clicking 'Upload PDF'.")
    
    st.markdown("---")

def render_agent_settings_section():
    """
    Renders the section for agent settings, including the web search toggle.
    Updates the 'web_search_enabled' flag in session state.
    """
    st.header("⚙️ Agent Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Web search toggle
        web_search_enabled = st.checkbox(
            "🌐 Enable Web Search", 
            value=st.session_state.web_search_enabled,
            help="When enabled, the agent can search the web if document knowledge is insufficient."
        )
        st.session_state.web_search_enabled = web_search_enabled
        
        # Show web search status
        if web_search_enabled:
            st.success("🌐 Web search is **enabled**")
        else:
            st.warning("📚 Only using **uploaded documents**")
    
    with col2:
        # Session controls
        if st.button("🔄 New Session"):
            from session_manager import reset_session
            reset_session()
            st.success("✅ New session started!")
            st.rerun()
        
        if st.button("📊 Session Stats"):
            stats = get_session_stats()
            st.info(f"""
            **Session Statistics:**
            - Messages: {stats['message_count']}
            - Uploads: {stats['upload_count']}
            - Web Search: {'On' if stats['web_search_enabled'] else 'Off'}
            - Errors: {stats['consecutive_errors']}
            """)
    
    st.markdown("---")

def display_chat_history():
    """Displays all messages currently in the session state chat history."""
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add timestamp for recent messages
            if i >= len(st.session_state.messages) - 5:  # Last 5 messages
                st.caption(f"_{time.strftime('%H:%M:%S')}_")

def display_trace_events(trace_events: list):
    """
    Renders the detailed agent workflow trace in an expandable section.
    Uses icons and conditional styling for better readability.
    """
    if not trace_events:
        return
        
    with st.expander(f"🔬 Agent Workflow Trace ({len(trace_events)} steps)", expanded=False):
        # Summary of the workflow
        nodes_used = [event['node_name'] for event in trace_events if event['node_name'] != '__end__']
        unique_nodes = list(dict.fromkeys(nodes_used))  # Preserve order, remove duplicates
        
        st.info(f"**Workflow Path:** {' → '.join(unique_nodes)}")
        
        # Detailed trace
        for event in trace_events:
            icon_map = {
                'router': "🎯",
                'rag_lookup': "📚",
                'web_search': "🌐",
                'answer': "💡",
                '__end__': "✅"
            }
            icon = icon_map.get(event['node_name'], "⚙️")
            
            # Create expandable section for each step
            with st.expander(f"{icon} Step {event['step']}: {event['node_name'].title()}", expanded=False):
                st.write(f"**Description:** {event['description']}")
                
                # Special handling for different node types
                if event['node_name'] == 'rag_lookup' and 'sufficiency_verdict' in event['details']:
                    verdict = event['details']['sufficiency_verdict']
                    if verdict == "Sufficient":
                        st.success(f"**RAG Verdict:** {verdict} ✅")
                        st.write("📚 Found relevant information in uploaded documents.")
                    else:
                        st.warning(f"**RAG Verdict:** {verdict} ⚠️")
                        st.write("🔍 Insufficient information found, proceeding to web search.")
                    
                    if 'retrieved_content_summary' in event['details']:
                        with st.expander("📄 Retrieved Content Preview"):
                            st.code(event['details']['retrieved_content_summary'], language=None)
                
                elif event['node_name'] == 'web_search' and 'retrieved_content_summary' in event['details']:
                    st.success("🌐 Web search completed successfully")
                    with st.expander("🔍 Web Search Results Preview"):
                        st.code(event['details']['retrieved_content_summary'], language=None)
                
                elif event['node_name'] == 'router':
                    decision = event['details'].get('decision', 'Unknown')
                    st.info(f"**Routing Decision:** {decision}")
                    
                    if 'override_reason' in event['details']:
                        st.warning(f"**Override Reason:** {event['details']['override_reason']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Initial:** {event['details'].get('initial_decision', 'N/A')}")
                        with col2:
                            st.write(f"**Final:** {event['details'].get('final_decision', 'N/A')}")
                
                elif event['node_name'] == 'answer':
                    st.success("💡 Generating final response using gathered context")
                
                elif event['node_name'] == '__end__':
                    st.success("✅ Agent workflow completed successfully")
                
                # Show raw details if available and not already displayed
                if event['details'] and event['node_name'] not in ['rag_lookup', 'web_search', 'router']:
                    with st.expander("🔧 Technical Details"):
                        st.json(event['details'])
        
        # Performance summary
        total_steps = len(trace_events)
        processing_time = "~1-2 minutes"  # Estimate
        
        st.success(f"""
        **Workflow Summary:**
        - Total Steps: {total_steps}
        - Processing Time: {processing_time}
        - Status: Completed Successfully ✅
        """)

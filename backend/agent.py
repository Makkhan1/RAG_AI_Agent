# rag_agent_app/backend/agent.py

import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# FIXED: Correct Tavily import
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    try:
        from langchain_tavily import TavilySearchResults
        TAVILY_AVAILABLE = True
    except ImportError:
        TAVILY_AVAILABLE = False
        print("Warning: Tavily search not available")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .config import GROQ_API_KEY, TAVILY_API_KEY
from .vectorstore import get_retriever, check_vectorstore_health

# Set API Keys as environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_AVAILABLE and TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# FIXED: Initialize Tavily search tool
if TAVILY_AVAILABLE and TAVILY_API_KEY:
    web_search_tool = TavilySearchResults(max_results=3)
else:
    web_search_tool = None
    print("Warning: Web search disabled - Tavily not available or API key missing")

# Define the state structure
class AgentState(Dict[str, Any]):
    messages: list
    route: str = ""
    rag: str = ""
    web: str = ""
    rag_verdict_is_sufficient: bool = False
    initial_router_decision: str = ""
    router_override_reason: str = ""
    web_search_enabled: bool = True

# Router node
def router(state: AgentState) -> AgentState:
    """Route the conversation to either RAG lookup or direct answer."""
    print("--- Entering router_node ---")
    
    messages = state["messages"]
    last_message = messages[-1].content
    
    print(f"Router analyzing query: {last_message}")
    
    try:
        # Check if vectorstore is healthy
        is_healthy, health_msg = check_vectorstore_health()
        
        if is_healthy:
            decision = "rag"
            print(f"Router decision: {decision} (vectorstore is healthy)")
        else:
            print(f"Vectorstore not healthy: {health_msg}")
            decision = "web" if state.get("web_search_enabled", True) and web_search_tool else "answer"
            print(f"Router decision: {decision} (bypassing unhealthy vectorstore)")
            
    except Exception as e:
        print(f"Router error: {e}")
        decision = "web" if state.get("web_search_enabled", True) and web_search_tool else "answer"
        print(f"Router decision: {decision} (error fallback)")
    
    state["route"] = decision
    state["initial_router_decision"] = decision
    
    print(f"--- Exiting router_node with route: {decision} ---")
    return state

# RAG lookup node
def rag_lookup(state: AgentState) -> AgentState:
    """Retrieve relevant documents and determine if sufficient for answering."""
    print("--- Entering rag_node ---")
    
    query = state["messages"][-1].content
    print(f"RAG query: {query}")
    
    try:
        # Get retriever
        retriever = get_retriever()
        
        # Retrieve relevant documents
        rag_docs = retriever.get_relevant_documents(query)
        
        if rag_docs:
            # Combine all retrieved content
            rag_content = "\n\n".join([doc.page_content for doc in rag_docs])
            state["rag"] = rag_content
            
            print(f"Retrieved {len(rag_docs)} documents from RAG")
            
            # Create a prompt to determine if the content is sufficient
            sufficiency_prompt = f"""
            Based on the following retrieved content, can you answer the user's question: "{query}"?
            
            Retrieved Content:
            {rag_content[:2000]}...
            
            Respond with only "SUFFICIENT" if the content contains enough information to answer the question,
            or "NOT_SUFFICIENT" if more information is needed.
            """
            
            try:
                # Ask LLM if content is sufficient
                sufficiency_response = llm.invoke([HumanMessage(content=sufficiency_prompt)])
                verdict = sufficiency_response.content.strip().upper()
                
                if "SUFFICIENT" in verdict:
                    state["rag_verdict_is_sufficient"] = True
                    state["route"] = "answer"
                    print("RAG content judged as SUFFICIENT. Proceeding to answer.")
                else:
                    state["rag_verdict_is_sufficient"] = False
                    if state.get("web_search_enabled", True) and web_search_tool:
                        state["route"] = "web"
                        print("RAG content judged as NOT SUFFICIENT. Proceeding to web search.")
                    else:
                        state["route"] = "answer"
                        print("RAG content judged as NOT SUFFICIENT. Web search disabled. Proceeding to answer anyway.")
                        
            except Exception as e:
                print(f"Error in sufficiency check: {e}. Defaulting to NOT SUFFICIENT.")
                state["rag_verdict_is_sufficient"] = False
                state["route"] = "web" if state.get("web_search_enabled", True) and web_search_tool else "answer"
        else:
            print("No documents retrieved from RAG")
            state["rag"] = ""
            state["rag_verdict_is_sufficient"] = False
            state["route"] = "web" if state.get("web_search_enabled", True) and web_search_tool else "answer"
            
    except Exception as e:
        print(f"RAG Error: RAG_ERROR::{e}. Checking web search enabled status.")
        
        # Set empty RAG content and route to web search or answer
        state["rag"] = ""
        state["rag_verdict_is_sufficient"] = False
        
        if state.get("web_search_enabled", True) and web_search_tool:
            state["route"] = "web"
            print("Routing to web search due to RAG error.")
        else:
            state["route"] = "answer"
            print("Routing to answer due to RAG error and web search disabled.")
    
    print("--- Exiting rag_node ---")
    return state

# FIXED: Web search node with proper error handling
def web_search(state: AgentState) -> AgentState:
    """Perform web search to gather additional information."""
    print("--- Entering web_node ---")
    
    query = state["messages"][-1].content
    print(f"Web search query: {query}")
    
    if not web_search_tool:
        print("Web search tool not available")
        state["web"] = "Web search is not available"
        state["route"] = "answer"
        print("--- Exiting web_node (no tool available) ---")
        return state
    
    try:
        # FIXED: Use the correct method call for TavilySearchResults
        web_results = web_search_tool.invoke({"query": query})
        
        # Format the results
        if isinstance(web_results, list) and web_results:
            formatted_results = []
            for result in web_results:
                if isinstance(result, dict):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content")
                    url = result.get("url", "No URL")
                    formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
                else:
                    formatted_results.append(str(result))
            
            web_content = "\n\n".join(formatted_results)
        else:
            web_content = str(web_results)
        
        state["web"] = web_content
        state["route"] = "answer"
        
        print(f"Web snippets retrieved: {web_content[:200]}...")
        
    except Exception as e:
        print(f"Web search error: {e}")
        state["web"] = f"Web search failed: {e}"
        state["route"] = "answer"
    
    print("--- Exiting web_node ---")
    return state

# FIXED: Answer generation node with better response handling
def answer_node(state: AgentState) -> AgentState:
    """Generate the final answer using available context."""
    print("--- Entering answer_node ---")
    
    query = state["messages"][-1].content
    rag_content = state.get("rag", "")
    web_content = state.get("web", "")
    
    print(f"Generating answer for query: {query}")
    print(f"RAG content available: {bool(rag_content)}")
    print(f"Web content available: {bool(web_content)}")
    
    # Build context from available sources
    context_parts = []
    
    if rag_content:
        context_parts.append(f"Document Knowledge Base:\n{rag_content}")
    
    if web_content and not web_content.startswith("Web search failed") and web_content != "Web search is not available":
        context_parts.append(f"Web Search Results:\n{web_content}")
    
    if context_parts:
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Please provide a clear and helpful answer to the user's question using the context below.

Context:
---
{context}
---

Question: {query}

Answer:"""

    else:
        # Handle simple greetings and basic questions
        if query.lower().strip() in ["hi", "hello", "hey", "how are you", "what's up"]:
            prompt = f"""The user greeted you with: "{query}"

Please respond with a friendly greeting and let them know how you can help them. You can:
- Answer questions about uploaded documents
- Search the web for current information
- Provide explanations on various topics

Give a warm, helpful response."""
        else:
            prompt = f"""The user asked: "{query}"

Unfortunately, I don't have access to relevant information in my knowledge base for this specific question, and web search didn't return useful results.

Please provide a helpful response that:
1. Acknowledges their question
2. Explains that you need more context or documents
3. Suggests they can upload relevant documents or try rephrasing their question
4. Offers to help with other questions

Be friendly and helpful."""

    print(f"Sending prompt to LLM (first 200 chars): {prompt[:200]}...")
    
    try:
        # Generate response
        response = llm.invoke([HumanMessage(content=prompt)])
        final_answer = response.content
        
        # FIXED: Ensure we have a valid response
        if not final_answer or final_answer.strip() == "":
            final_answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question or ask me something else."
        
        print(f"Generated answer (first 200 chars): {final_answer[:200]}...")
        
        # FIXED: Add the AI response to messages - this is crucial!
        state["messages"].append(AIMessage(content=final_answer))
        print(f"Added AI message to state. Total messages now: {len(state['messages'])}")
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while generating a response: {str(e)}"
        print(f"Error in answer generation: {e}")
        state["messages"].append(AIMessage(content=error_msg))
    
    print("--- Exiting answer_node ---")
    return state

# Conditional routing function
def route_question(state: AgentState) -> str:
    """Determine the next node based on the current route."""
    route = state.get("route", "")
    print(f"Routing to: {route}")
    
    if route == "rag":
        return "rag_lookup"
    elif route == "web":
        return "web_search"
    elif route == "answer":
        return "answer"
    else:
        print(f"Unknown route: {route}, defaulting to answer")
        return "answer"

# Build the graph
def create_agent_graph():
    """Create and return the agent graph."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("rag_lookup", rag_lookup)
    workflow.add_node("web_search", web_search)
    workflow.add_node("answer", answer_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "rag_lookup": "rag_lookup",
            "web_search": "web_search", 
            "answer": "answer"
        }
    )
    
    workflow.add_conditional_edges(
        "rag_lookup",
        route_question,
        {
            "web_search": "web_search",
            "answer": "answer"
        }
    )
    
    # Direct edges to end
    workflow.add_edge("web_search", "answer")
    workflow.add_edge("answer", END)
    
    # Compile the graph with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Create the agent
rag_agent = create_agent_graph()

# Test function
def test_agent():
    """Test the agent with a simple query."""
    try:
        config = {"configurable": {"thread_id": "test_thread"}}
        inputs = {"messages": [HumanMessage(content="Hello, can you help me?")]}
        
        for step in rag_agent.stream(inputs, config=config):
            print(f"Step: {step}")
            
        return True
    except Exception as e:
        print(f"Agent test failed: {e}")
        return False

if __name__ == "__main__":
    test_agent()

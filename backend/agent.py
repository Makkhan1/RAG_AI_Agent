import os
from typing import List, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# --- ABSOLUTE IMPORTS ---
from config import GEMINI_API_KEY, TAVILY_API_KEY
from vectorstore import get_retriever, pc, INDEX_NAME, embeddings

# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=2, topic="general")

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"

@tool
def rag_search_tool(query: str, config: RunnableConfig) -> str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        # --- FIX: Re-enable Session Isolation ---
        # We extract the session_id (passed as thread_id) and use it as the namespace.
        # This ensures we only search documents belonging to THIS session.
        session_id = config.get("configurable", {}).get("thread_id")
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, 
            embedding=embeddings,
            namespace=session_id  # Search ONLY within this session's data
        )
        retriever_instance = vectorstore.as_retriever()
        
        docs = retriever_instance.invoke(query, k=10)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"

# --- Pydantic schemas for structured output ---
class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, False otherwise.")

# --- GEMINI LLM SETUP WITH FALLBACK ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in environment variables.")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Set max_retries=1 to fail fast and trigger the fallback model immediately
gemini_primary = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    max_retries=1
)

gemini_secondary = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=1
)

gemini_tertiary = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=3
)

llm_with_fallback = gemini_primary.with_fallbacks([gemini_secondary, gemini_tertiary])

router_llm = llm_with_fallback.with_structured_output(RouteDecision)
judge_llm = llm_with_fallback.with_structured_output(RagJudge)
answer_llm = llm_with_fallback

# --- Shared state type ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool
    rag_verdict_is_sufficient: bool
    has_documents: bool

# --- Node 1: router (decision) ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering router_node (Gemini) ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    # Extract configuration
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    session_id = config.get("configurable", {}).get("thread_id")
    
    # --- CHECK PINECONE STATS (SESSION SPECIFIC) ---
    has_documents = False
    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        
        # --- FIX: Check Specific Namespace ---
        # We only check if documents exist in the namespace matching the current session_id.
        if "namespaces" in stats and session_id in stats["namespaces"]:
            if stats["namespaces"][session_id].vector_count > 0:
                has_documents = True
        
        print(f"Session {session_id} Stats: has_docs={has_documents}")
    except Exception as e:
        print(f"Warning: Could not check index stats: {e}. Defaulting to False.")
    
    print(f"Router config: web_search={web_search_enabled}, has_documents={has_documents}")

    # 2. CONSTRUCT SYSTEM PROMPT
    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
    )

    if has_documents:
        system_prompt += (
            "\nPrioritize using the **internal knowledge base (RAG)** for factual information, "
            "summaries, and questions about the document's content (e.g., 'What is this about?', 'Main topic?', 'Summarize')."
        )
    else:
        system_prompt += "\n**NOTICE: The Knowledge Base is currently EMPTY.** You CANNOT use the 'rag' route."

    if web_search_enabled:
        system_prompt += (
            "\nYou **CAN** use web search for queries that require very current, real-time, or broad general knowledge."
            "\nRoutes: 'rag', 'web', 'answer', 'end'."
        )
        if not has_documents:
             system_prompt += "\nSince KB is empty, if the user asks for information, default to 'web'."
    else:
        system_prompt += (
            "\n**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "\nRoutes: 'rag', 'answer', 'end'."
        )
        if not has_documents:
             system_prompt += "\nSince KB is empty and Web is disabled, you must use 'answer' and rely on your own knowledge."

    system_prompt += (
        "\n'answer': For simple direct questions or math."
        "\n'end': For greetings/small talk. Must provide a reply."
    )

    messages = [("system", system_prompt), ("user", query)]
    
    result: RouteDecision = router_llm.invoke(messages)
    
    initial_router_decision = result.route
    router_override_reason = None

    # Logic Override: If LLM chooses RAG but we know it's empty
    if result.route == "rag" and not has_documents:
        if web_search_enabled:
            result.route = "web"
            router_override_reason = "Knowledge Base is empty; redirected to Web Search."
        else:
            result.route = "answer"
            router_override_reason = "Knowledge Base is empty and Web is disabled; redirected to LLM Answer."
        print(f"Router decision overridden: {router_override_reason}")

    # Logic Override: If User disabled Web but LLM chose Web
    if not web_search_enabled and result.route == "web":
        result.route = "rag" if has_documents else "answer"
        router_override_reason = "Web search disabled by user; redirected to fallback."
        print(f"Router decision overridden: changed from 'web' to '{result.route}' because web search is disabled.")
    
    out = {
        "messages": state["messages"], 
        "route": result.route,
        "web_search_enabled": web_search_enabled,
        "has_documents": has_documents,
        # --- FIX: Clear Context at start of every turn ---
        "rag": "", 
        "web": "",
        "rag_verdict_is_sufficient": False
    }
    if router_override_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]
    
    print("--- Exiting router_node ---")
    return out

# --- Node 2: RAG lookup ---
def rag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering rag_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    
    # Invoke tool with config to pass thread_id/namespace
    chunks = rag_search_tool.invoke(query, config=config)
    
    # 1. EMPTY RAG CHECK (Optimization)
    if not chunks or chunks.startswith("RAG_ERROR::"):
        print("RAG returned no chunks or error. Skipping judge.")
        next_route = "web" if web_search_enabled else "answer"
        return {
            **state, 
            "rag": "", 
            "route": next_route,
            "rag_verdict_is_sufficient": False
        }

    # 2. JUDGE
    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
    ]
    verdict: RagJudge = judge_llm.invoke(judge_messages)
    
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer"

    print("--- Exiting rag_node ---")
    return {
        **state,
        "rag": chunks,
        "route": next_route,
        "rag_verdict_is_sufficient": verdict.sufficient,
        "web_search_enabled": web_search_enabled
    }

# --- Node 3: web search ---
def web_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    
    if not web_search_enabled:
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    snippets = web_search_tool.invoke(query)
    
    if snippets.startswith("WEB_ERROR::"):
        return {**state, "web": "", "route": "answer"}

    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}

# --- Node 4: final answer ---
def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    ctx_parts = []
    # Note: State is now fresh, so these will only exist if populated in THIS turn
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        if state["web"] and not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
    
    context = "\n\n".join(ctx_parts).strip()
    
    if not context:
        # Check if we should refuse based on user settings (No Docs + No Web)
        has_documents = state.get("has_documents", False)
        web_search_enabled = state.get("web_search_enabled", True)
        
        if not has_documents and not web_search_enabled:
            print("Blocking answer from general knowledge as per policy (No Docs + No Web).")
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content="Please upload a document to the Knowledge Base or turn on Web Search to proceed.")]
            }

        prompt = f"""You are a helpful assistant. 
The user asked a question, but you have no access to external tools (Web/RAG) or they returned no results.
Please answer the question to the best of your ability using your general knowledge.

Question: {user_q}
"""
    else:
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
Context:
---
{context}
---
Question: {user_q}
"""

    ans = answer_llm.invoke([HumanMessage(content=prompt)]).content
    print("--- Exiting answer_node ---")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }

# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

# --- Build graph ---
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")
    
    g.add_conditional_edges("router", from_router, {"rag": "rag_lookup", "web": "web_search", "answer": "answer", "end": END})
    g.add_conditional_edges("rag_lookup", after_rag, {"answer": "answer", "web": "web_search"})
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    agent = g.compile(checkpointer=MemorySaver())
    return agent

rag_agent = build_agent()

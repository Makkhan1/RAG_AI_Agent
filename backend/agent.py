import os
from typing import List, Literal, TypedDict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq  # Official Groq Integration
from langchain_tavily import TavilySearch
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# --- ABSOLUTE IMPORTS ---
from config import GROQ_API_KEY, TAVILY_API_KEY
from vectorstore import get_retriever, pc, INDEX_NAME, embeddings

# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=5, search_depth="advanced")

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
        # --- Session Isolation ---
        session_id = config.get("configurable", {}).get("thread_id")
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, 
            embedding=embeddings,
            namespace=session_id  # Search ONLY within this session's data
        )
        retriever_instance = vectorstore.as_retriever()
        
        docs = retriever_instance.invoke(query, k=5)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"

# --- Pydantic schemas for structured output ---
class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., description="True if retrieved information is sufficient.")

# --- GROQ LLM SETUP WITH 3-MODEL FALLBACK ---
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in environment variables.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Primary Model
primary_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0,
    max_retries=1
)

# Secondary Fallback
secondary_llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0,
    max_retries=1
)

# Tertiary Fallback
tertiary_llm = ChatGroq(
    model_name="openai/gpt-oss-safeguard-20b",
    temperature=0,
    max_retries=3
)

# Combine into a single fallback chain
llm_with_fallback = primary_llm.with_fallbacks([secondary_llm, tertiary_llm])

# Prepare Structured Models
router_llm = llm_with_fallback.with_structured_output(RouteDecision,method="json_mode")
judge_llm = llm_with_fallback.with_structured_output(RagJudge,method="json_mode")
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
    initial_router_decision: str
    router_override_reason: str

# --- Node 1: router (decision) ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering router_node (Groq) ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    session_id = config.get("configurable", {}).get("thread_id")
    
    # --- CHECK PINECONE STATS (SESSION SPECIFIC) ---
    has_documents = False
    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        if "namespaces" in stats and session_id in stats["namespaces"]:
            if stats["namespaces"][session_id].vector_count > 0:
                has_documents = True
        print(f"Session {session_id} Stats: has_docs={has_documents}")
    except Exception as e:
        print(f"Warning: Could not check index stats: {e}. Defaulting to False.")

    # CONSTRUCT SYSTEM PROMPT
    system_prompt = (
        "You are an RAG AI agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "You MUST respond with a valid JSON object. " 
        "Direct user queries to: 'rag', 'web', 'answer', or 'end'."
    )

    if has_documents:
        system_prompt += (
        "\nCRITICAL: The user has uploaded documents. For ANY question about the "
        "content, topic, summary, or specific details of the uploaded file, "
        "you MUST use the 'rag' route. Do not attempt to answer from memory."
    )
    else:
        system_prompt += "\n**NOTICE: The Knowledge Base is currently EMPTY.** You CANNOT use the 'rag' route."

    if web_search_enabled:
        system_prompt += "\nYou **CAN** use web search. Default to 'web' if KB is empty."
    else:
        system_prompt += "\n**Web search is DISABLED.** Use 'answer' if KB is empty."

    system_prompt += "\n'answer': simple direct questions. 'end': greetings/small talk (provide a reply)."

    messages = [("system", system_prompt), ("user", query)]
    
    result: RouteDecision = router_llm.invoke(messages)
    
    initial_router_decision = result.route
    router_override_reason = None

    # Logic Override: KB Empty
    if result.route == "rag" and not has_documents:
        if web_search_enabled:
            result.route = "web"
            router_override_reason = "Knowledge Base is empty; redirected to Web Search."
        else:
            result.route = "answer"
            router_override_reason = "Knowledge Base is empty and Web is disabled; redirected to LLM Answer."

    # Logic Override: Web Disabled
    if not web_search_enabled and result.route == "web":
        result.route = "rag" if has_documents else "answer"
        router_override_reason = "Web search disabled by user; redirected to fallback."
    
    out = {
        "messages": state["messages"], 
        "route": result.route,
        "web_search_enabled": web_search_enabled,
        "has_documents": has_documents,
        "rag": "", 
        "web": "",
        "rag_verdict_is_sufficient": False
    }
    
    if router_override_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason
        print(f"Router decision overridden: {router_override_reason}")

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]
    
    print("--- Exiting router_node ---")
    return out

# --- Node 2: RAG lookup ---
def rag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering rag_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = state.get("web_search_enabled", True)
    
    chunks = rag_search_tool.invoke(query, config=config)
    
    if not chunks or chunks.startswith("RAG_ERROR::"):
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route, "rag_verdict_is_sufficient": False}

    judge_messages = [
        ("system", "Respond ONLY with JSON: {\"sufficient\": true/false}"),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient?")
    ]
    verdict: RagJudge = judge_llm.invoke(judge_messages)
    
    next_route = "answer" if verdict.sufficient else ("web" if web_search_enabled else "answer")

    print("--- Exiting rag_node ---")
    return {
        **state,
        "rag": chunks,
        "route": next_route,
        "rag_verdict_is_sufficient": verdict.sufficient
    }

# --- Node 3: web search ---
def web_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    if not state.get("web_search_enabled", True):
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    snippets = web_search_tool.invoke(query)
    web_content = "" if snippets.startswith("WEB_ERROR::") else snippets

    print("--- Exiting web_node ---")
    return {**state, "web": web_content, "route": "answer"}

# --- Node 4: final answer ---
def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        if not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
    
    context = "\n\n".join(ctx_parts).strip()
    
    if not context:
        if not state.get("has_documents", False) and not state.get("web_search_enabled", True):
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content="Please upload a document or turn on Web Search to proceed.")]
            }
        prompt = f"Answer using general knowledge. Question: {user_q}"
    else:
        prompt = f"Answer using ONLY context:\n\n{context}\n\nQuestion: {user_q}"

    ans_msg = answer_llm.invoke([HumanMessage(content=prompt)])
    
    print("--- Exiting answer_node ---")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans_msg.content)]
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

    return g.compile(checkpointer=MemorySaver())

rag_agent = build_agent()

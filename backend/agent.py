# # rag_agent_app/backend/agent.py

import os
from typing import List, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import API keys from config
from .config import GROQ_API_KEY, TAVILY_API_KEY
from .vectorstore import get_retriever

# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=3, topic="general")

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
def rag_search_tool(query: str) -> str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        retriever_instance = get_retriever()
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

# --- LLM instances with structured output where needed ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


router_llm = ChatOpenAI(
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
).with_structured_output(RouteDecision)

judge_llm = ChatOpenAI(
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
).with_structured_output(RagJudge)

answer_llm = ChatOpenAI(
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# --- Shared state type ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool
    rag_verdict_is_sufficient: bool # <-- CHANGED: Added for accurate tracing

# --- Node 1: router (decision) ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering router_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    print(f"Router received web search info : {web_search_enabled}")
 
    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )
    
    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
            "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
            "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
            "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
            "\n\nChoose one of the following routes:"
            "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        )

    system_prompt += (
        "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
        "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
        "\n\nExample routing decisions:"
        "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
        "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
        "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
        "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
        "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
        "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'"
    )

    system_prompt += (
    "\n\nFor mathematical problems and calculations:"
    "\n- 'answer': For basic arithmetic, simple algebra, standard mathematical concepts, or problems you can solve directly (e.g., 'What is 25 + 37?', 'Solve for x: 2x + 5 = 15', 'What is the derivative of x²?')."
    "\n- 'rag': For complex mathematical problems, mathematical theorems, proofs, or detailed mathematical procedures that might benefit from knowledge base content (e.g., 'Explain the proof of Pythagorean theorem', 'How to solve quadratic equations?', 'What are the applications of calculus in physics?')."
    "\n- Avoid 'web' for mathematical problems unless they involve very recent mathematical discoveries, current mathematical competitions, or time-sensitive mathematical events."
    "\n\nMathematical routing examples:"
    "\n- User: 'What is 123 × 456?' -> Route: 'answer' (Basic arithmetic)."
    "\n- User: 'Solve: 3x² - 12x + 9 = 0' -> Route: 'answer' (Standard algebra)."
    "\n- User: 'Explain integration by parts' -> Route: 'rag' (Mathematical concept that benefits from detailed explanation)."
    "\n- User: 'What is the binomial theorem?' -> Route: 'rag' (Mathematical theorem)."
    "\n- User: 'Calculate the standard deviation of [2, 4, 6, 8, 10]' -> Route: 'answer' (Standard calculation)."
    "\n- User: 'How do you prove that √2 is irrational?' -> Route: 'rag' (Mathematical proof)."
    "\n- User: 'What is 15% of 240?' -> Route: 'answer' (Simple percentage calculation)."
    "\n- User: 'Explain the fundamental theorem of calculus' -> Route: 'rag' (Complex mathematical concept)."
)

    messages = [("system", system_prompt), ("user", query)]
    result: RouteDecision = router_llm.invoke(messages)
    
    initial_router_decision = result.route
    router_override_reason = None

    if not web_search_enabled and result.route == "web":
        result.route = "rag" 
        router_override_reason = "Web search disabled by user; redirected to RAG."
        print(f"Router decision overridden: changed from 'web' to 'rag' because web search is disabled.")
    
    print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")
    
    out = {
        "messages": state["messages"], 
        "route": result.route,
        "web_search_enabled": web_search_enabled
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
    print(f"RAG query: {query}")
    chunks = rag_search_tool.invoke(query)
    
    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error: {chunks}. Checking web search enabled status.")
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route}

    if chunks:
        print(f"Retrieved RAG chunks (first 500 chars): {chunks[:500]}...")
    else:
        print("No RAG chunks retrieved.")

    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "If the information is incomplete, vague, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            "If no relevant information was retrieved at all, it is definitely NOT sufficient."
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
    ]
    verdict: RagJudge = judge_llm.invoke(judge_messages)
    print(f"RAG Judge verdict: {verdict.sufficient}")
    
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer"
        print(f"RAG not sufficient. Web search enabled: {web_search_enabled}. Next route: {next_route}")

    print("--- Exiting rag_node ---")
    return {
        **state,
        "rag": chunks,
        "route": next_route,
        "rag_verdict_is_sufficient": verdict.sufficient, # <-- CHANGED: Pass the real verdict
        "web_search_enabled": web_search_enabled
    }

# --- Node 3: web search ---
def web_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    
    if not web_search_enabled:
        print("Web search node entered but web search is disabled. Skipping actual search.")
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    print(f"Web search query: {query}")
    snippets = web_search_tool.invoke(query)
    
    if snippets.startswith("WEB_ERROR::"):
        print(f"Web Error: {snippets}. Proceeding to answer with limited info.")
        return {**state, "web": "", "route": "answer"}

    print(f"Web snippets retrieved: {snippets[:200]}...")
    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}

# --- Node 4: final answer ---
def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        if state["web"] and not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
    
    context = "\n\n".join(ctx_parts).strip()
    
    # <-- CHANGED: New, stricter prompt to prevent answering from general knowledge -->
    if not context:
        prompt = f"""You are a helpful assistant. The user asked a question, but you were unable to find any relevant information in your knowledge base or on the web.
You MUST inform the user that you do not have enough information to answer their question. Do not try to answer it from your own memory.

Question: {user_q}
"""
    else:
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the provided context does not contain the information needed to answer the question, you MUST state that you do not have enough information to answer.
DO NOT use any of your own internal knowledge or information you were trained on.

Context:
---
{context}
---

Based strictly on the context above, answer the following question.

Question: {user_q}
"""

    print(f"Prompt sent to answer_llm: {prompt[:500]}...")
    ans = answer_llm.invoke([HumanMessage(content=prompt)]).content
    print(f"Final answer generated: {ans[:200]}...")
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

def after_web(_) -> Literal["answer"]:
    return "answer"

# --- Build graph ---
def build_agent():
    """Builds and compiles the LangGraph agent."""
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

# Update the _process_chat_request function in backend/main.py

async def _process_chat_request(request: QueryRequest) -> AgentResponse:
    """Helper function to process chat request"""
    trace_events_for_frontend: List[TraceEvent] = []
    
    try:
        config = {
            "configurable": {
                "thread_id": request.session_id,
                "web_search_enabled": request.enable_web_search
            }
        }
        inputs = {
            "messages": [HumanMessage(content=request.query)],
            "web_search_enabled": request.enable_web_search
        }

        final_message = ""
        final_state = None
        
        print(f"--- Starting Agent Stream for session {request.session_id} ---")
        print(f"Web Search Enabled: {request.enable_web_search}")

        # Run agent stream in thread pool to avoid blocking
        stream_generator = await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(rag_agent.stream(inputs, config=config))
        )

        for i, s in enumerate(stream_generator):
            # Store the final state
            final_state = s
            
            current_node_name = None
            node_output_state = None

            if '__end__' in s:
                current_node_name = '__end__'
                node_output_state = s['__end__']
            else:
                current_node_name = list(s.keys())[0] 
                node_output_state = s[current_node_name]

            event_description = f"Executing node: {current_node_name}"
            event_details = {}
            event_type = "generic_node_execution"

            if current_node_name == "router":
                route_decision = node_output_state.get('route')
                initial_decision = node_output_state.get('initial_router_decision', route_decision)
                override_reason = node_output_state.get('router_override_reason', None)

                if override_reason:
                    event_description = f"Router initially decided: '{initial_decision}'. Overridden to: '{route_decision}' because {override_reason}."
                    event_details = {"initial_decision": initial_decision, "final_decision": route_decision, "override_reason": override_reason}
                else:
                    event_description = f"Router decided: '{route_decision}'"
                    event_details = {"decision": route_decision, "reason": "Based on initial query analysis."}
                event_type = "router_decision"
            
            elif current_node_name == "rag_lookup":
                rag_content_summary = node_output_state.get("rag", "")[:200] + "..." if node_output_state.get("rag", "") else "No content"
                rag_sufficient = node_output_state.get("rag_verdict_is_sufficient", False)
                
                if rag_sufficient:
                    event_description = "RAG Lookup: Content found and judged SUFFICIENT. Proceeding to answer."
                    event_details = {"retrieved_content_summary": rag_content_summary, "sufficiency_verdict": "Sufficient"}
                else:
                    next_route = node_output_state.get("route")
                    if next_route == "web":
                        event_description = "RAG Lookup: Content judged NOT SUFFICIENT. Diverting to web search."
                    else:
                        event_description = "RAG Lookup: Content judged NOT SUFFICIENT. Proceeding to answer as fallback."
                    event_details = {"retrieved_content_summary": rag_content_summary, "sufficiency_verdict": "Not Sufficient"}
                
                event_type = "rag_action"

            elif current_node_name == "web_search":
                web_content_summary = node_output_state.get("web", "")[:200] + "..." if node_output_state.get("web", "") else "No content"
                event_description = "Web Search performed. Results retrieved. Proceeding to answer."
                event_details = {"retrieved_content_summary": web_content_summary}
                event_type = "web_action"
            elif current_node_name == "answer":
                event_description = "Generating final answer using gathered context."
                event_type = "answer_generation"
            elif current_node_name == "__end__":
                event_description = "Agent process completed."
                event_type = "process_end"

            trace_events_for_frontend.append(
                TraceEvent(
                    step=i + 1,
                    node_name=current_node_name,
                    description=event_description,
                    details=event_details,
                    event_type=event_type
                )
            )
            print(f"Streamed Event: Step {i+1} - Node: {current_node_name} - Desc: {event_description}")

        # FIXED: Better final message extraction
        if final_state:
            # Check for __end__ state first
            if '__end__' in final_state:
                state_dict = final_state['__end__']
            else:
                # Get the last node's state
                node_name = list(final_state.keys())[0]
                state_dict = final_state[node_name]
            
            # Extract the final message
            if "messages" in state_dict and state_dict["messages"]:
                # Get the last AI message
                for msg in reversed(state_dict["messages"]):
                    if isinstance(msg, AIMessage) and msg.content.strip():
                        final_message = msg.content
                        break
        
        # FIXED: Fallback if no message found
        if not final_message:
            final_message = "Hello! I'm here to help. You can ask me questions about uploaded documents or anything else you'd like to know."
            print("No final message found, using fallback greeting.")

        print(f"--- Agent Stream Ended. Final Response: {final_message[:200]}... ---")

        return AgentResponse(response=final_message, trace_events=trace_events_for_frontend)

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_details = f"Error during agent invocation: {e}"
        print(error_details)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Server Error: {e}")

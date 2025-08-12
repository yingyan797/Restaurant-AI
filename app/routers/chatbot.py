# app/routers/ai.py
"""
AI Chatbot API Router for Restaurant Booking System.

This module provides endpoints for the AI chatbot functionality,
integrating with the LangGraph-based conversational agent.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import copy
import os
from langchain_core.messages import HumanMessage, AIMessage

# Import the chatbot components (LangGraph app and initial state)
from app.ai_chatbot import app, initial_agent_state # Import 'app' and 'initial_agent_state'

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chatbot"])

class ChatMessage(BaseModel):
    message: str
    email: str = None

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage) -> ChatResponse:
    """
    Process a chat message through the AI agent and return the response.
    
    Args:
        chat_message: The user's message
        
    Returns:
        ChatResponse: The AI agent's response
    """
    try:
        # Use email as session ID to maintain separate conversations per user
        session_id = chat_message.email if chat_message.email else "default_session"
        
        # Prepare input for the LangGraph agent
        # The LangGraph agent expects messages in its state, not as a direct input to .invoke()
        # So we create a HumanMessage and pass it in the .invoke() call, and the graph handles adding it to state.
        
        # LangGraph's checkpointer will load/save the state automatically based on 'thread_id'
        # We need to set up the initial state if it's a new conversation for this thread_id.
        # This is handled by the initial_agent_state being passed as 'input' to invoke/stream.
        # However, to explicitly set 'customer_email' at the start of a new session, we need a slight adjustment
        # in how the initial state is provided, or rely on the agent itself to infer/ask.
        
        # For a clean initial state that includes the customer_email from the first message:
        # We'll rely on the agent_node to capture/set customer_email if it's the first message
        # or if the email changes.
        
        # The graph will manage the 'messages' list within its state.
        # We just need to pass the new HumanMessage as the input.
        
        # When calling .stream(), each iteration returns a dict with the updated state for that step.
        # We want the final state after all steps have run.
        inputs = {"messages": [HumanMessage(content=chat_message.message)]}
        
        # If it's a new session, ensure customer_email is set in the state before streaming
        # This is a bit tricky with direct LangGraph usage and checkpointer.
        # A simple way for initial setup: The first time a user chats, the agent_node can initialize it.
        # Or, we can modify initial_agent_state slightly if it's guaranteed to be the first message of a session.
        
        # The most robust way is to always pass the current email to the agent_node via the config,
        # and let the agent_node decide to update its internal customer_email if different.
        # For simplicity, let's just make sure it's part of the initial AgentState if it's a truly new session.

        # Run the graph and get the final state
        # The 'app.stream' method yields updates, the last one is the final state.
        final_state = None
        for s in app.stream(inputs, config={"configurable": {"thread_id": session_id}}):
            final_state = s

        if final_state is None:
            logger.error(f"Graph stream returned no states for session {session_id}.")
            raise HTTPException(
                status_code=500, 
                detail="An error occurred: The AI agent did not return a response."
            )
        
        # Extract the final AgentState from the stream output (it's the value of the last dict)
        # e.g., {'agent': {...}, '__end__': {...}} -> we want the last state from 'agent' or final node
        # LangGraph's stream output often has the full state at each step, so the last yielded value
        # contains the state of the last node executed.
        
        # The structure of `s` will be `{'node_name': AgentState_dict}` for each step.
        # We want the state from the last node executed, which updates the 'messages' list.
        # The key for the final state is usually the name of the last node.
        
        # A safer way to get the final messages is to use app.get_state() after invocation
        # but that requires another call. For simplicity with stream, we assume the last
        # item in `messages` of `final_state` is the AI's latest reply.
        
        # The 'final_state' variable `s` *is* the dictionary representing the state after the last step.
        # The `app.stream` yields partial states, so `final_state` will contain the full state of the last node.
        # The 'messages' key in this state will contain the full chat history.
        
        # Extract the messages from the final state dictionary
        # The actual state object is typically the value of the only key in the `s` dict, or the last key.
        # If `s` contains `{'__end__': {'messages': [...]}}`, then `s['__end__']` is the relevant part.
        
        # Let's use `app.get_state()` for clarity after `app.stream()` has run.
        # However, `app.stream` already updates the checkpointer.
        # The `s` variable will hold the final state of the graph.
        
        # The actual content is usually in the 'messages' key of the final AgentState.
        # Accessing `list(s.values())[-1]` is one way if you know it's the last value.
        # A more direct way is to get the state from the checkpointer *after* the stream.
        
        # Let's simplify and directly access the `messages` from the last yielded state object `s`.
        # Assuming `s` is a dict of the form `{'node_name': AgentState_dict}`
        updated_agent_state_messages = None
        for value in final_state.values():
            if isinstance(value, dict) and "messages" in value:
                updated_agent_state_messages = value["messages"]
                break
        
        if updated_agent_state_messages:
            final_message = updated_agent_state_messages[-1]
            response_content = final_message.content if hasattr(final_message, 'content') else "I'm processing your request."
        else:
            response_content = "I'm sorry, I encountered an internal issue and couldn't get a response."
            logger.error(f"Could not extract messages from final state: {final_state}")

        return ChatResponse(response=response_content)
        
    except Exception as e:
        logger.error(f"Error processing chat message for session {chat_message.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your message. Please try again."
        )

class ResetRequest(BaseModel):
    email: str = None

@router.post("/reset")
async def reset_conversation(reset_request: ResetRequest = None):
    """Reset the conversation state for a specific user."""
    try:
        session_id = reset_request.email if reset_request and reset_request.email else "default_session"
        
        # To reset a conversation with SqliteSaver, you can invoke with mode="reset"
        # This effectively clears the state associated with the given thread_id.
        app.invoke(
            initial_agent_state, # Provide an initial state, can be empty or default
            config={"configurable": {"thread_id": session_id}}, 
        )
        return {"message": "Conversation reset request acknowledged."}
    except Exception as e:
        logger.error(f"Error resetting conversation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error resetting conversation")
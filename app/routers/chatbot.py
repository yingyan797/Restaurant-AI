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

# Import the chatbot components (LangGraph app, initial state, and memory_saver)
from app.ai_chatbot import app, initial_agent_state, memory_saver # Import 'memory_saver'

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
        
        inputs = {
            "messages": [HumanMessage(content=chat_message.message)],
            "customer_email": chat_message.email
        }
        
        final_state = None
        for s in app.stream(inputs, config={"configurable": {"thread_id": session_id}, "recursion_limit": 5}):
            final_state = s

        if final_state is None:
            logger.error(f"Graph stream returned no states for session {session_id}.")
            raise HTTPException(
                status_code=500, 
                detail="An error occurred: The AI agent did not return a response."
            )
        
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
        
        # Clear the state for the given thread_id in the checkpointer
        # This effectively clears the state associated with the given thread_id.
        memory_saver.delete_thread(session_id)

        return {"message": "Conversation reset successfully. You can start a new conversation."}
    except Exception as e:
        logger.error(f"Error resetting conversation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error resetting conversation")
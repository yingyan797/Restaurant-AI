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
from langchain_core.messages import HumanMessage

# Set up environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyAnKWcDA8jjl_Rgf1gJcdm_UBNB2UzWHXo"

# Import the chatbot components
from app.ai_chatbot import graph, initial_agent_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chatbot"])

class ChatMessage(BaseModel):
    message: str
    email: str = None

class ChatResponse(BaseModel):
    response: str

# Store conversation states per session (in production, use Redis or database)
conversation_states: Dict[str, Any] = {}

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
        session_id = chat_message.email or "default_session"
        
        # Get or create conversation state
        if session_id not in conversation_states:
            conversation_states[session_id] = copy.deepcopy(initial_agent_state)
            if chat_message.email:
                conversation_states[session_id]["user_email"] = chat_message.email
        
        current_state = conversation_states[session_id]
        
        # Update email if changed (triggers reset)
        if chat_message.email and current_state["user_email"] != chat_message.email:
            conversation_states[session_id] = copy.deepcopy(initial_agent_state)
            conversation_states[session_id]["user_email"] = chat_message.email
            current_state = conversation_states[session_id]
        
        # Add user message to state
        current_state['messages'].append(HumanMessage(content=chat_message.message))
        
        # Clear previous state for new interaction
        current_state["current_tool_call"] = None
        current_state["pending_questions"] = []
        
        # Reset some state if no action is pending (fresh start)
        # This logic ensures a fresh start for a new high-level intent,
        # but preserves context within an ongoing action.
        if current_state["requested_action"] is None:
            current_state["bookings_summary"] = []
            current_state["selected_booking_reference"] = None
            current_state["cancellation_reasons"] = []
            current_state["microsite_name"] = None
            current_state["extracted_params"] = {}
        
        # Process through the graph
        last_step_output = None # Renamed from final_state for clarity
        for step_state_dict in graph.stream(current_state):
            # Each step_state_dict is like {'node_name': {updated_AgentState_dict}}
            last_step_output = step_state_dict
            
        # Extract the actual AgentState from the last yielded item
        if last_step_output:
            # Get the value of the single key-value pair, which is the full AgentState
            updated_agent_state = list(last_step_output.values())[0] 
        else:
            logger.error("Graph stream returned no states in chat_endpoint.")
            # Fallback to current_state if no output from stream, possibly resetting or erroring
            updated_agent_state = copy.deepcopy(initial_agent_state)
            if chat_message.email:
                updated_agent_state["user_email"] = chat_message.email
            updated_agent_state['messages'].append(HumanMessage(content="I'm sorry, I encountered an internal issue. Please try again."))

        # Update stored state with the extracted AgentState
        conversation_states[session_id] = updated_agent_state
        
        # Get the AI's response from the updated_agent_state
        final_message = updated_agent_state['messages'][-1]
        
        if hasattr(final_message, 'content'):
            response_content = final_message.content
        else:
            # Fallback if content attribute is missing (e.g., malformed message type)
            response_content = "I'm processing your request. Please provide more information if needed."
        
        return ChatResponse(response=response_content)
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your message. Please try again."
        )

class ResetRequest(BaseModel):
    email: str = None

@router.post("/reset")
async def reset_conversation(reset_request: ResetRequest = None):
    """Reset the conversation state."""
    try:
        session_id = reset_request.email if reset_request and reset_request.email else "default_session"
        new_state = copy.deepcopy(initial_agent_state)
        if reset_request and reset_request.email:
            new_state["user_email"] = reset_request.email
        conversation_states[session_id] = new_state
        return {"message": "Conversation reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail="Error resetting conversation")
# app/routers/ai.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from langchain_core.messages import HumanMessage, AIMessage
from app.ai_chatbot import app, initial_agent_state, AgentState, memory

# NEW: Import 'memory' (the checkpointer instance) from ai_chatbot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["AI Chatbot"])

class ChatMessage(BaseModel):
    message: str
    email: str = None

class ChatResponse(BaseModel):
    response: str

# OLD: Remove the global in-memory store for conversation states
# _user_conversation_states: Dict[str, AgentState] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage) -> ChatResponse:
    """
    Process a chat message through the AI agent and return the response.
    
    Args:
        chat_message: The user's message
        
    Returns:
        ChatResponse: The AI agent's response
    """

    session_id = chat_message.email if chat_message.email else "default_session"
    current_state = initial_agent_state
    current_state.user_message = chat_message.message
    current_state.selected_email = chat_message.email or ""
    
    try:
        print("Enter/Reenter loop from", current_state.entry_point)
        final_state = None
        response_content = ""
        node_sequence = []
        for s in app.stream(current_state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 10}):
            final_state = s
            node_sequence.append(s.keys())
            for key in s.keys():
                if key == "pause_or_exit":
                    response_content = s[key].get("agent_response")
                    for k, v in s[key].items():
                        if k == "entry_point":
                            if v == "execute_tool_action":
                                initial_agent_state.entry_point = None
                            elif v == "extract_additional_pause":
                                initial_agent_state.entry_point = "extract_additional_resume"
                            else:
                                initial_agent_state.entry_point = v
                        else:
                            setattr(initial_agent_state, k, v)
                    break
            else:
                continue
            break
            
        print(node_sequence)
        if final_state is None:
            logger.error(f"Graph stream returned no states for session {session_id}.")
            raise HTTPException(
                status_code=500, 
                detail="An error occurred: The AI agent did not return a response."
            )
        
        if not response_content:
            response_content = "I'm sorry, I encountered an internal issue and couldn't get a response."
            logger.error(f"Could not extract messages from final state")

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
    session_id = reset_request.email if reset_request and reset_request.email else "default_session"
    
    try:
        # NEW: Use the compiled graph's method to delete the thread state
        session_id = reset_request.email if reset_request and reset_request.email else "default_session"
        
        # Clear the state for the given thread_id in the checkpointer
        # This effectively clears the state associated with the given thread_id.
        memory.delete_thread(session_id)
        initial_agent_state.reset()
        return {"message": "Conversation reset successfully. You can start a new conversation."}
    except Exception as e:
        logger.error(f"Error resetting conversation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error resetting conversation")
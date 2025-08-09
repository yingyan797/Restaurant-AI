"""
AI Agents Router for Restaurant Booking Chatbot.

Provides mock AI response endpoints for the chatbot functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/ai", tags=["AI Agents"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat_response(request: ChatRequest) -> ChatResponse:
    """Generate AI chatbot response for user messages."""
    message = request.message.lower()
    
    if any(word in message for word in ['availability', 'available', 'check', 'free']):
        response = "I can help you check availability! Please provide the date and party size, or use the 'Check Availability' section."
    elif any(word in message for word in ['book', 'reservation', 'reserve', 'table']):
        response = "I'd be happy to help you make a booking! Please provide your preferred date, time, party size, and contact details."
    elif any(word in message for word in ['cancel', 'modify', 'change', 'update']):
        response = "I can help you manage your booking. Please provide your booking reference number."
    elif any(word in message for word in ['hello', 'hi', 'help', 'start']):
        response = "Hello! I'm your restaurant booking assistant. I can help you:\n• Check table availability\n• Make new bookings\n• Manage existing reservations\n\nWhat would you like to do?"
    elif any(word in message for word in ['time', 'hours', 'open']):
        response = "We're open for lunch (12:00-14:00) and dinner (19:00-21:00). What time would you prefer?"
    elif any(word in message for word in ['menu', 'food', 'cuisine']):
        response = "I focus on booking assistance, but I can help you secure a table to enjoy our cuisine! Would you like to make a reservation?"
    else:
        response = "I can help you with restaurant bookings. Try asking about availability, making a booking, or managing an existing reservation!"
    
    return ChatResponse(response=response)
"""
AI Agents Router for Restaurant Booking Chatbot.

Provides LangGraph workflow for handling restaurant booking and availability requests.
"""

from typing import Dict, Any, List, Optional, TypedDict
from datetime import date, time, datetime
import re

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from app.database import get_db
from app.ai_tools import (
    query_table, get_customer_information, check_customer_bookings_and_restaurants,
    QueryConfig, FilterCondition, FilterOperator, create_equals_filter
)

router = APIRouter(prefix="/api/ai", tags=["AI Agents"])

class ChatRequest(BaseModel):
    message: str
    email: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    action_needed: Optional[str] = None
    missing_info: Optional[List[str]] = None

class WorkflowState(TypedDict):
    message: str
    email: Optional[str]
    intent: Optional[str]
    extracted_info: Dict[str, Any]
    missing_info: List[str]
    response: str
    action_needed: Optional[str]
    api_call_needed: bool

def analyze_intent(state: WorkflowState) -> WorkflowState:
    """Analyze user message to determine intent."""
    message = state["message"].lower()
    
    if any(word in message for word in ['availability', 'available', 'check', 'free', 'slots']):
        state["intent"] = "check_availability"
    elif any(word in message for word in ['book', 'reservation', 'reserve', 'table', 'make']):
        state["intent"] = "make_booking"
    elif any(word in message for word in ['cancel', 'cancellation']):
        state["intent"] = "cancel_booking"
    elif any(word in message for word in ['modify', 'change', 'update', 'edit']):
        state["intent"] = "update_booking"
    elif any(word in message for word in ['my booking', 'booking details', 'find booking', 'get booking']):
        state["intent"] = "get_booking"
    elif any(word in message for word in ['my bookings', 'all bookings', 'list bookings']):
        state["intent"] = "list_bookings"
    else:
        state["intent"] = "general_help"
    
    return state

def extract_information(state: WorkflowState) -> WorkflowState:
    """Extract relevant information from user message."""
    message = state["message"]
    extracted = {}
    
    # Extract date patterns
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(today|tomorrow|next week)'
    ]
    for pattern in date_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            extracted["date"] = match.group(1)
            break
    
    # Extract time patterns
    time_match = re.search(r'(\d{1,2}:?\d{0,2}\s*(?:am|pm|AM|PM)?)', message)
    if time_match:
        extracted["time"] = time_match.group(1)
    
    # Extract party size
    party_patterns = [
        r'(\d+)\s*(?:people|person|guests?|pax)',
        r'party\s*(?:of|size)?\s*(\d+)',
        r'for\s*(\d+)'
    ]
    for pattern in party_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            extracted["party_size"] = int(match.group(1))
            break
    
    # Extract booking reference
    ref_match = re.search(r'([A-Z0-9]{7})', message)
    if ref_match:
        extracted["booking_reference"] = ref_match.group(1)
    
    # Extract email
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', message)
    if email_match:
        extracted["email"] = email_match.group(1)
    
    state["extracted_info"] = extracted
    return state

def check_missing_info(state: WorkflowState) -> WorkflowState:
    """Check what information is missing for the intended action."""
    intent = state["intent"]
    extracted = state["extracted_info"]
    missing = []
    
    if intent == "check_availability":
        if "date" not in extracted:
            missing.append("visit date")
        if "party_size" not in extracted:
            missing.append("party size")
    
    elif intent == "make_booking":
        if "date" not in extracted:
            missing.append("visit date")
        if "time" not in extracted:
            missing.append("visit time")
        if "party_size" not in extracted:
            missing.append("party size")
        if not state.get("email") and "email" not in extracted:
            missing.append("email address")
    
    elif intent in ["cancel_booking", "update_booking", "get_booking"]:
        if "booking_reference" not in extracted:
            missing.append("booking reference")
    
    elif intent == "list_bookings":
        if not state.get("email") and "email" not in extracted:
            missing.append("email address")
    
    state["missing_info"] = missing
    state["api_call_needed"] = len(missing) == 0 and intent != "general_help"
    return state

def handle_api_call(state: WorkflowState) -> WorkflowState:
    """Handle API calls based on intent and available information."""
    intent = state["intent"]
    extracted = state["extracted_info"]
    
    try:
        if intent == "check_availability":
            visit_date = extracted["date"]
            party_size = extracted["party_size"]
            
            db = next(get_db())
            from app.models import Restaurant, AvailabilitySlot, Booking
            
            restaurant = db.query(Restaurant).filter(Restaurant.name == "TheHungryUnicorn").first()
            if restaurant:
                slots = db.query(AvailabilitySlot).filter(
                    AvailabilitySlot.restaurant_id == restaurant.id,
                    AvailabilitySlot.date == visit_date,
                    AvailabilitySlot.max_party_size >= party_size
                ).all()
                
                available_times = []
                for slot in slots:
                    existing_bookings = db.query(Booking).filter(
                        Booking.restaurant_id == restaurant.id,
                        Booking.visit_date == visit_date,
                        Booking.visit_time == slot.time,
                        Booking.status == "confirmed"
                    ).count()
                    
                    if slot.available and existing_bookings < 3:
                        available_times.append(slot.time.strftime("%H:%M"))
                
                if available_times:
                    state["response"] = f"Available times for {party_size} people on {visit_date}: {', '.join(available_times)}"
                else:
                    state["response"] = f"Sorry, no availability for {party_size} people on {visit_date}."
            else:
                state["response"] = "Restaurant not found."
        
        elif intent == "list_bookings":
            email = state.get("email") or extracted.get("email")
            result = check_customer_bookings_and_restaurants(
                email,
                QueryConfig(columns=["booking_reference", "visit_date", "visit_time", "party_size", "status"]),
                QueryConfig(columns=["name"])
            )
            
            if result.success and result.data:
                bookings_text = "\n".join([
                    f"• {b['booking_reference']}: {b['visit_date']} at {b['visit_time']} for {b['party_size']} people ({b['status']})"
                    for b in result.data
                ])
                state["response"] = f"Your bookings:\n{bookings_text}"
            else:
                state["response"] = "No bookings found for this email address."
        
        elif intent == "get_booking":
            booking_ref = extracted["booking_reference"]
            booking_config = QueryConfig(
                filters=[create_equals_filter("booking_reference", booking_ref)],
                limit=1
            )
            result = query_table("bookings", booking_config)
            
            if result.success and result.data:
                booking = result.data[0]
                state["response"] = f"Booking {booking_ref}: {booking['visit_date']} at {booking['visit_time']} for {booking['party_size']} people. Status: {booking['status']}"
            else:
                state["response"] = f"Booking {booking_ref} not found."
        
        else:
            state["response"] = "This action requires using the booking form or API endpoints directly."
            state["action_needed"] = "use_api_endpoint"
    
    except Exception as e:
        state["response"] = f"Error processing request: {str(e)}"
    
    return state

def generate_response(state: WorkflowState) -> WorkflowState:
    """Generate appropriate response based on state."""
    if state["missing_info"]:
        missing_list = ", ".join(state["missing_info"])
        state["response"] = f"To help you with that, I need the following information: {missing_list}"
        state["action_needed"] = "collect_info"
    elif state["intent"] == "general_help":
        state["response"] = "Hello! I can help you with:\n• Check table availability\n• View your bookings\n• Find booking details\n\nFor making, updating, or canceling bookings, please use the API endpoints directly.\n\nWhat would you like to do?"
    elif not state.get("response"):
        state["response"] = "I understand you want to " + state["intent"].replace("_", " ") + ". Let me help you with that."
    
    return state

def should_call_api(state: WorkflowState) -> str:
    """Determine if API call is needed."""
    return "api_call" if state["api_call_needed"] else "generate_response"

# Create the workflow graph
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("extract_information", extract_information)
workflow.add_node("check_missing_info", check_missing_info)
workflow.add_node("handle_api_call", handle_api_call)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("analyze_intent", "extract_information")
workflow.add_edge("extract_information", "check_missing_info")
workflow.add_conditional_edges(
    "check_missing_info",
    should_call_api,
    {
        "api_call": "handle_api_call",
        "generate_response": "generate_response"
    }
)
workflow.add_edge("handle_api_call", "generate_response")
workflow.add_edge("generate_response", END)

# Set entry point
workflow.set_entry_point("analyze_intent")

# Compile the graph
app_workflow = workflow.compile()

@router.post("/chat", response_model=ChatResponse)
async def chat_response(request: ChatRequest) -> ChatResponse:
    """Process user message through LangGraph workflow."""
    initial_state = {
        "message": request.message,
        "email": request.email,
        "intent": None,
        "extracted_info": {},
        "missing_info": [],
        "response": "",
        "action_needed": None,
        "api_call_needed": False
    }
    
    # Run the workflow
    final_state = app_workflow.invoke(initial_state)
    
    return ChatResponse(
        response=final_state["response"],
        action_needed=final_state.get("action_needed"),
        missing_info=final_state["missing_info"] if final_state["missing_info"] else None
    )
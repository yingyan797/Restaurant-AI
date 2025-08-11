from datetime import date, time, datetime
from typing import Dict, Any, List, Optional, Callable

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from app.database import get_db, Base, engine
from app.models import Restaurant, Customer, Booking, AvailabilitySlot, CancellationReason
from ai_tools import AIToolCallingInterface, logger

# --- LangGraph State Definition ---
# This defines the schema of information that will be passed between nodes in the graph.
class AgentState(TypedDict):
    current_intent: Optional[str]  # e.g., "SearchAvailabilityTool", "CreateBookingTool", "ask_for_info", "respond_directly"
    user_message: str
    tool_output: Optional[Any]  # The raw output from the tool execution
    chat_history: List[Dict[str, Any]]  # Stores {"role": "user/assistant", "content": "message"} for context
    missing_info: List[str]  # List of parameters still needed for a tool call
    response: Optional[str]  # The final response to the user
    original_tool_intent: Optional[str] # Stores the tool name if we branched to ask_for_info

    # Parameters for various API calls, flattened for simplicity and direct access
    restaurant_name: Optional[str]
    VisitDate: Optional[date]  # Stored as date object internally after parsing
    VisitTime: Optional[time]  # Stored as time object internally after parsing
    PartySize: Optional[int]
    ChannelCode: Optional[str]  # Default "ONLINE"
    SpecialRequests: Optional[str]
    IsLeaveTimeConfirmed: Optional[bool]
    RoomNumber: Optional[str]
    booking_reference: Optional[str]  # For get, cancel, update booking
    micrositeName: Optional[str]  # For cancellation, often same as restaurant name
    cancellationReasonId: Optional[int]  # For cancellation

    # Customer fields (for booking creation and customer lookup)
    customer_email: Optional[str]
    customer_first_name: Optional[str]
    customer_surname: Optional[str]
    customer_mobile: Optional[str]
    customer_title: Optional[str]
    customer_mobile_country_code: Optional[str]
    customer_phone_country_code: Optional[str]
    customer_phone: Optional[str]
    customer_receive_email_marketing: Optional[bool]
    customer_receive_sms_marketing: Optional[bool]
    customer_group_email_marketing_opt_in_text: Optional[str]
    customer_group_sms_marketing_opt_in_text: Optional[str]
    customer_receive_restaurant_email_marketing: Optional[bool]
    customer_receive_restaurant_sms_marketing: Optional[bool]
    customer_restaurant_email_marketing_opt_in_text: Optional[str]
    customer_restaurant_sms_marketing_opt_in_text: Optional[str]

# --- Pydantic Models for Tool Definitions (for Gemini) ---
# These models describe the structure of the inputs for each tool, which Gemini uses
# to understand how to call them.

class DateStr(str):
    """Custom Pydantic type for date strings in YYYY-MM-DD format."""
    @classmethod
    def __get_validators__(cls): yield cls.validate
    @classmethod
    def validate(cls, v):
        try: date.fromisoformat(v); return v
        except ValueError: raise ValueError("Date must be in YYYY-MM-DD format")

class TimeStr(str):
    """Custom Pydantic type for time strings in HH:MM:SS format."""
    @classmethod
    def __get_validators__(cls): yield cls.validate
    @classmethod
    def validate(cls, v):
        try: time.fromisoformat(v); return v
        except ValueError: raise ValueError("Time must be in HH:MM:SS format")

class SearchAvailabilityTool(BaseModel):
    """Search for available booking slots at a restaurant for a specific date and party size."""
    restaurant_name: str = Field(description="The name of the restaurant (e.g., 'The Fancy Fork').")
    VisitDate: DateStr = Field(description="The desired visit date in YYYY-MM-DD format (e.g., '2025-12-25').")
    PartySize: int = Field(description="Number of people in the party (e.g., 2).")
    ChannelCode: str = Field(default="ONLINE", description="The booking channel identifier (default 'ONLINE').")

class CreateBookingTool(BaseModel):
    """Create a new restaurant booking for a customer."""
    restaurant_name: str = Field(description="The name of the restaurant.")
    VisitDate: DateStr = Field(description="The desired visit date in YYYY-MM-DD format.")
    VisitTime: TimeStr = Field(description="The desired visit time in HH:MM:SS format (e.g., '19:00:00').")
    PartySize: int = Field(description="Number of people in the party.")
    customer_email: str = Field(description="Customer's email address.")
    customer_first_name: str = Field(description="Customer's first name.")
    customer_surname: str = Field(description="Customer's surname.")
    customer_mobile: str = Field(description="Customer's mobile number.")
    ChannelCode: str = Field(default="ONLINE", description="The booking channel identifier (default 'ONLINE').")
    SpecialRequests: Optional[str] = Field(None, description="Any special requests for the booking (e.g., 'table by window', 'allergy to nuts').")
    IsLeaveTimeConfirmed: Optional[bool] = Field(False, description="Whether the leave time is confirmed (boolean).")
    RoomNumber: Optional[str] = Field(None, description="Room number if applicable (e.g., hotel guest).")
    customer_title: Optional[str] = Field(None, description="Customer's title (e.g., 'Mr', 'Ms', 'Dr').")
    customer_mobile_country_code: Optional[str] = Field(None, description="Customer's mobile country code (e.g., '+1').")
    customer_phone_country_code: Optional[str] = Field(None, description="Customer's phone country code.")
    customer_phone: Optional[str] = Field(None, description="Customer's phone number.")
    customer_receive_email_marketing: Optional[bool] = Field(False, description="Customer opts in to receive email marketing (boolean).")
    customer_receive_sms_marketing: Optional[bool] = Field(False, description="Customer opts in to receive SMS marketing (boolean).")
    customer_group_email_marketing_opt_in_text: Optional[str] = Field(None, description="Text for group email marketing opt-in.")
    customer_group_sms_marketing_opt_in_text: Optional[str] = Field(None, description="Text for group SMS marketing opt-in.")
    customer_receive_restaurant_email_marketing: Optional[bool] = Field(False, description="Customer opts in to receive restaurant specific email marketing (boolean).")
    customer_receive_restaurant_sms_marketing: Optional[bool] = Field(False, description="Customer opts in to receive restaurant specific SMS marketing (boolean).")
    customer_restaurant_email_marketing_opt_in_text: Optional[str] = Field(None, description="Text for restaurant email marketing opt-in.")
    customer_restaurant_sms_marketing_opt_in_text: Optional[str] = Field(None, description="Text for restaurant SMS marketing opt-in.")

class CancelBookingTool(BaseModel):
    """Cancel an existing restaurant booking."""
    restaurant_name: str = Field(description="The name of the restaurant where the booking was made.")
    booking_reference: str = Field(description="The unique booking reference to cancel.")
    micrositeName: str = Field(description="The name of the microsite, usually the same as the restaurant name.")
    cancellationReasonId: int = Field(description="The ID of the reason for cancellation. Use list_cancellation_reasons tool to find valid IDs.")

class GetBookingDetailsTool(BaseModel):
    """Get detailed information about an existing restaurant booking."""
    restaurant_name: str = Field(description="The name of the restaurant.")
    booking_reference: str = Field(description="The unique booking reference.")

class UpdateBookingDetailsTool(BaseModel):
    """Update details of an existing restaurant booking. At least one updatable field must be provided."""
    restaurant_name: str = Field(description="The name of the restaurant.")
    booking_reference: str = Field(description="The unique booking reference.")
    VisitDate: Optional[DateStr] = Field(None, description="New desired visit date in YYYY-MM-DD format.")
    VisitTime: Optional[TimeStr] = Field(None, description="New desired visit time in HH:MM:SS format.")
    PartySize: Optional[int] = Field(None, description="New number of people in the party.")
    SpecialRequests: Optional[str] = Field(None, description="New special requests for the booking.")
    IsLeaveTimeConfirmed: Optional[bool] = Field(None, description="New status for leave time confirmation (boolean).")

class GetCustomerBookingsAndRestaurantsSummaryTool(BaseModel):
    """Get a summary of a customer's bookings and the associated restaurant information."""
    email: str = Field(description="The customer's email address.")

class ListCancellationReasonsTool(BaseModel):
    """Get all possible cancellation reasons with their IDs, titles, and descriptions."""

class GetRestaurantsTool(BaseModel):
    """Get information of all restaurants or a specific restaurant by name."""
    name: Optional[str] = Field(None, description="Optional: The name of the specific restaurant to search for. If omitted, lists all restaurants.")

# --- LangGraph Nodes ---

# Initialize a Gemini model with tool calling capabilities
# Set your GOOGLE_API_KEY environment variable or uncomment the line below to set it directly
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
model = ChatGoogleGenerativeAI(model="gemini-pro")

# Bind the defined tools to the Gemini model
model_with_tools = model.bind_tools([
    SearchAvailabilityTool, CreateBookingTool, CancelBookingTool, GetBookingDetailsTool,
    UpdateBookingDetailsTool, GetCustomerBookingsAndRestaurantsSummaryTool,
    ListCancellationReasonsTool, GetRestaurantsTool
])

def _map_state_to_tool_params(state: AgentState, tool_model: BaseModel) -> Dict[str, Any]:
    """
    Maps relevant fields from the AgentState to the parameters expected by a given tool's Pydantic model.
    Handles type conversions (e.g., date/time objects to strings) where necessary.
    """
    params = {}
    for field_name, field_info in tool_model.model_fields.items():
        state_value = getattr(state, field_name, None)
        
        # Convert date/time objects back to string format if the tool expects strings
        if field_name == "VisitDate" and isinstance(state_value, date):
            params[field_name] = state_value.isoformat()
        elif field_name == "VisitTime" and isinstance(state_value, time):
            params[field_name] = state_value.isoformat()
        elif state_value is not None:
            params[field_name] = state_value

    return params

def agent_node(state: AgentState) -> AgentState:
    """
    The main decision-making node. It uses Gemini to determine user intent,
    extract parameters, and decide the next action (call tool, ask for info, or respond directly).
    """
    logger.info("Entering agent_node...")
    messages = [SystemMessage("You are a helpful assistant for restaurant bookings. "
                              "You can search for availability, create, cancel, update, and retrieve bookings. "
                              "You can also list restaurants and cancellation reasons. "
                              "Always ask for all necessary information before calling a tool. "
                              "Be polite and informative. "
                              "Current Date: " + date.today().isoformat() + "\n"
                              "Current Time: " + datetime.now().strftime("%H:%M:%S") + "\n"
                              "User can provide dates in YYYY-MM-DD format and times in HH:MM:SS format.")]

    # Add relevant chat history for context
    for msg in state['chat_history']:
        messages.append(AIMessage(content=msg["content"]) if msg["role"] == "assistant" else HumanMessage(content=msg["content"]))

    # Add the current user message as the last HumanMessage for Gemini to process
    messages.append(HumanMessage(content=state['user_message']))

    response = model_with_tools.invoke(messages)
    logger.info(f"Gemini raw response: {response}")

    new_state = state.copy()
    new_state['tool_output'] = None
    new_state['missing_info'] = []
    new_state['response'] = None

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call.name
        tool_args = tool_call.args

        # Preserve the original tool intent if we are coming from an "ask_for_info" state
        if new_state.get('current_intent') == "ask_for_info" and new_state.get('original_tool_intent'):
            new_state['current_intent'] = new_state['original_tool_intent']
        else:
            new_state['current_intent'] = tool_name
            new_state['original_tool_intent'] = tool_name # Store the tool name as the original intent

        logger.info(f"Gemini suggested tool: {tool_name} with args: {tool_args}")

        # Update state with extracted arguments, performing type conversions for date/time
        for key, value in tool_args.items():
            if key == "VisitDate" and isinstance(value, str):
                try: new_state[key] = date.fromisoformat(value)
                except ValueError: logger.warning(f"Could not parse VisitDate: {value}"); new_state[key] = None
            elif key == "VisitTime" and isinstance(value, str):
                try: new_state[key] = time.fromisoformat(value)
                except ValueError: logger.warning(f"Could not parse VisitTime: {value}"); new_state[key] = None
            else:
                new_state[key] = value

        # Check for missing required parameters for the determined tool
        tool_model_map = {
            "SearchAvailabilityTool": SearchAvailabilityTool, "CreateBookingTool": CreateBookingTool,
            "CancelBookingTool": CancelBookingTool, "GetBookingDetailsTool": GetBookingDetailsTool,
            "UpdateBookingDetailsTool": UpdateBookingDetailsTool,
            "GetCustomerBookingsAndRestaurantsSummaryTool": GetCustomerBookingsAndRestaurantsSummaryTool,
            "ListCancellationReasonsTool": ListCancellationReasonsTool, "GetRestaurantsTool": GetRestaurantsTool,
        }
        target_tool_model = tool_model_map.get(new_state['current_intent'])

        if target_tool_model:
            required_fields = target_tool_model.schema().get('required', [])
            for field_name in required_fields:
                if new_state.get(field_name) is None:
                    new_state['missing_info'].append(field_name)
            
            # Special check for UpdateBookingDetailsTool: needs at least one updatable field
            if new_state['current_intent'] == "UpdateBookingDetailsTool" and not new_state['missing_info']:
                update_fields = ["VisitDate", "VisitTime", "PartySize", "SpecialRequests", "IsLeaveTimeConfirmed"]
                update_fields_present = any(new_state.get(f) is not None for f in update_fields)
                if not update_fields_present:
                    new_state['missing_info'].append("at least one of (VisitDate, VisitTime, PartySize, SpecialRequests, IsLeaveTimeConfirmed)")
        else:
            logger.warning(f"No Pydantic model found for tool: {new_state['current_intent']}")

        # Decide next state based on whether information is missing
        if new_state['missing_info']:
            new_state['current_intent'] = "ask_for_info"
        else:
            new_state['current_intent'] = "call_tool"
        
        return new_state
    else:
        # If Gemini doesn't suggest a tool call, it's a direct conversational response
        new_state['response'] = response.content
        new_state['current_intent'] = "respond_directly"
        new_state['original_tool_intent'] = None # Clear original intent
        return new_state

def tool_node(state: AgentState) -> AgentState:
    """
    Executes the tool chosen by the agent based on the current state's `current_intent`
    (which will be a tool name like "SearchAvailabilityTool").
    """
    logger.info("Entering tool_node...")
    tool_name = state['current_intent'] # This is the actual tool name to execute
    new_state = state.copy()

    if tool_name not in AIToolCallingInterface.TOOL_FUNCTIONS:
        new_state['response'] = f"Error: Internal system error, unknown tool '{tool_name}'."
        new_state['tool_output'] = {"error": "Unknown tool"}
        new_state['current_intent'] = "error"
        return new_state

    tool_func = AIToolCallingInterface.TOOL_FUNCTIONS[tool_name]
    tool_model_map = {
        "SearchAvailabilityTool": SearchAvailabilityTool, "CreateBookingTool": CreateBookingTool,
        "CancelBookingTool": CancelBookingTool, "GetBookingDetailsTool": GetBookingDetailsTool,
        "UpdateBookingDetailsTool": UpdateBookingDetailsTool,
        "GetCustomerBookingsAndRestaurantsSummaryTool": GetCustomerBookingsAndRestaurantsSummaryTool,
        "ListCancellationReasonsTool": ListCancellationReasonsTool, "GetRestaurantsTool": GetRestaurantsTool,
    }
    target_tool_model = tool_model_map.get(tool_name)
    if not target_tool_model:
        new_state['response'] = f"Error: Internal system error, no Pydantic model for '{tool_name}'."
        new_state['tool_output'] = {"error": "No tool model"}
        new_state['current_intent'] = "error"
        return new_state

    try:
        # Prepare arguments by mapping state to tool model fields
        args_for_tool = _map_state_to_tool_params(new_state, target_tool_model)
        logger.info(f"Executing tool {tool_name} with args: {args_for_tool}")
        tool_result = tool_func(**args_for_tool)
        new_state['tool_output'] = tool_result
        new_state['current_intent'] = tool_name # Keep the tool name for response generation
    except Exception as e:
        logger.error(f"Error during tool execution for {tool_name}: {e}")
        new_state['tool_output'] = {"error": str(e)}
        new_state['response'] = f"I encountered an error while trying to complete your request: {e}"
        new_state['current_intent'] = "error" # Change intent to error to trigger error response

    return new_state

def prepare_response_node(state: AgentState) -> AgentState:
    """
    Generates a user-friendly response based on the `current_intent` and `tool_output`.
    """
    logger.info("Entering prepare_response_node...")
    new_state = state.copy()

    # If a direct response was already set by agent_node (for conversational replies), use it
    if new_state.get('response') and new_state.get('current_intent') == "respond_directly":
        return new_state

    tool_output = new_state.get('tool_output')
    current_intent = new_state.get('current_intent')

    if tool_output and tool_output.get("error"):
        new_state['response'] = f"Sorry, there was an issue: {tool_output['error']}. Please try again or provide more details."
        return new_state
    
    response_message = "I'm not sure how to respond to that."

    # Construct response based on the tool that was executed
    if current_intent == "SearchAvailabilityTool":
        if tool_output and tool_output.get("available_slots"):
            slots = [s for s in tool_output["available_slots"] if s["available"]]
            if slots:
                slots_info = "\n".join([f"- {s['time']} (Max Party Size: {s['max_party_size']})" for s in slots])
                response_message = (
                    f"Available slots at {state['restaurant_name']} on {state['VisitDate']} for party size {state['PartySize']}:\n"
                    f"{slots_info}"
                )
            else:
                response_message = (
                    f"No available slots found for {state['restaurant_name']} on {state['VisitDate']} "
                    f"for party size {state['PartySize']}. Please try a different date or party size."
                )
        else: response_message = "I couldn't retrieve availability for the specified criteria."

    elif current_intent == "CreateBookingTool":
        if tool_output and tool_output.get("booking_reference"):
            response_message = (
                f"Booking confirmed for {tool_output['party_size']} people at {tool_output['restaurant']} "
                f"on {tool_output['visit_date']} at {tool_output['visit_time']}. "
                f"Your booking reference is: {tool_output['booking_reference']}. "
                f"A confirmation will be sent to {tool_output['customer']['email']}."
            )
        else: response_message = "I apologize, but I was unable to create your booking. Please check the details and try again."

    elif current_intent == "CancelBookingTool":
        if tool_output and tool_output.get("status") == "cancelled":
            response_message = (
                f"Your booking {tool_output['booking_reference']} at {tool_output['restaurant']} "
                f"has been successfully cancelled. Reason: {tool_output.get('cancellation_reason', 'N/A')}."
            )
        else: response_message = "I was unable to cancel your booking. Please ensure the booking reference and restaurant name are correct."

    elif current_intent == "GetBookingDetailsTool":
        if tool_output and tool_output.get("booking_id"):
            customer = tool_output.get("customer", {})
            cancellation_info = f"\nCancellation Reason: {tool_output['cancellation_reason']['reason']}" if tool_output.get("cancellation_reason") else ""
            response_message = (
                f"Booking Reference: {tool_output['booking_reference']}\n"
                f"Restaurant: {tool_output['restaurant']}\n"
                f"Date: {tool_output['visit_date']}, Time: {tool_output['visit_time']}\n"
                f"Party Size: {tool_output['party_size']}\n"
                f"Status: {tool_output['status']}\n"
                f"Customer: {customer.get('first_name')} {customer.get('surname')} ({customer.get('email')}, {customer.get('mobile')})\n"
                f"Special Requests: {tool_output.get('special_requests', 'None')}{cancellation_info}"
            )
        else: response_message = "I could not find a booking with that reference and restaurant name."

    elif current_intent == "UpdateBookingDetailsTool":
        if tool_output and tool_output.get("status") == "updated":
            updates = ", ".join([f"{k}: {v}" for k, v in tool_output.get('updates', {}).items()])
            response_message = (
                f"Booking {tool_output['booking_reference']} at {tool_output['restaurant']} has been successfully updated. "
                f"Changes: {updates}."
            )
        elif tool_output and tool_output.get("status") == "no_changes":
            response_message = f"Booking {tool_output['booking_reference']} was checked, but no changes were necessary."
        else: response_message = "I was unable to update your booking. Please try again."
        
    elif current_intent == "GetCustomerBookingsAndRestaurantsSummaryTool":
        if tool_output and isinstance(tool_output, dict) and tool_output.get("bookings_summary"):
            bookings_summary = tool_output.get("bookings_summary")
            booked_restaurants = tool_output.get("booked_restaurants")
            booking_list_str = "\n".join([f"- Ref: {b['booking_reference']} at {b['restaurant_name']}" for b in bookings_summary])
            restaurant_list_str = "\n".join([f"- {r['name']} (Address: {r['address']})" for r in booked_restaurants])
            response_message = (
                f"Here is a summary of bookings for {state.get('customer_email')}:\n"
                f"Your bookings:\n{booking_list_str}\n\n"
                f"Restaurants you've booked at:\n{restaurant_list_str}"
            )
        elif tool_output and tool_output.get("error"):
             response_message = tool_output["error"]
        else: response_message = f"No bookings found for {state.get('customer_email')}."

    elif current_intent == "ListCancellationReasonsTool":
        if tool_output and tool_output.get("success") and tool_output.get("data"):
            reasons_list = [f"- ID: {r['id']}, Reason: {r['reason']}" for r in tool_output["data"]]
            response_message = "Here are the available cancellation reasons:\n" + "\n".join(reasons_list)
        else: response_message = "I could not retrieve the list of cancellation reasons at this time."

    elif current_intent == "GetRestaurantsTool":
        if tool_output and tool_output.get("success") and tool_output.get("data"):
            if state.get('restaurant_name'): # If a specific restaurant was asked for
                r = tool_output["data"][0]
                response_message = (
                    f"Yes, I found '{r['name']}' located at {r['address']}. "
                    f"You can reach them at {r['phone_number']}."
                )
            else: # If all restaurants were requested
                restaurants_info = [f"- {r['name']} ({r['address']})" for r in tool_output["data"]]
                response_message = "Here are the restaurants I know:\n" + "\n".join(restaurants_info)
        else: response_message = "No restaurants found matching your criteria."

    new_state['response'] = response_message
    return new_state

def ask_for_info_node(state: AgentState) -> AgentState:
    """
    Generates a clarifying question to the user asking for the missing information.
    """
    logger.info("Entering ask_for_info_node...")
    new_state = state.copy()
    missing = new_state.get('missing_info', [])
    
    if not missing:
        new_state['response'] = "I'm ready to proceed, but it seems I'm not missing any information currently. How can I help?"
        return new_state

    # Customizable prompts for common missing fields
    prompts = {
        "restaurant_name": "the restaurant name", "VisitDate": "the visit date (e.g., YYYY-MM-DD)",
        "VisitTime": "the visit time (e.g., HH:MM:SS)", "PartySize": "the number of people in your party",
        "customer_email": "your email address", "customer_first_name": "your first name",
        "customer_surname": "your surname", "customer_mobile": "your mobile number",
        "booking_reference": "the booking reference number",
        "micrositeName": "the microsite name (usually the same as the restaurant name)",
        "cancellationReasonId": "the cancellation reason ID (you can ask me to list them by saying 'List cancellation reasons')",
        "at least one of (VisitDate, VisitTime, PartySize, SpecialRequests, IsLeaveTimeConfirmed)": "what you'd like to update (e.g., new date, time, party size, or special requests)",
    }

    # Format the list of missing items for the user-facing prompt
    missing_phrases = [
        prompts.get(item, item.replace('_', ' ').replace('customer ', 'your ').replace('IsLeaveTimeConfirmed', 'whether the leave time is confirmed'))
        for item in missing
    ]
    
    if len(missing_phrases) == 1:
        new_state['response'] = f"Could you please provide {missing_phrases[0]}?"
    elif len(missing_phrases) == 2:
        new_state['response'] = f"Could you please provide {missing_phrases[0]} and {missing_phrases[1]}?"
    else:
        last_item = missing_phrases.pop()
        new_state['response'] = f"Could you please provide {', '.join(missing_phrases)}, and {last_item}?"

    return new_state

# --- LangGraph Graph Definition ---
def create_booking_agent_graph():
    """
    Constructs and compiles the LangGraph state machine for the booking agent.
    """
    workflow = StateGraph(AgentState)

    # Add the nodes to the workflow
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", tool_node)
    workflow.add_node("prepare_response", prepare_response_node)
    workflow.add_node("ask_for_info", ask_for_info_node)

    # Set the initial entry point of the graph
    workflow.set_entry_point("agent")

    # Define conditional edges based on the 'current_intent' determined by the agent
    workflow.add_conditional_edges(
        "agent",
        lambda state: state['current_intent'],
        {
            "call_tool": "tool_executor",       # If a tool needs to be called
            "ask_for_info": "ask_for_info",     # If more info is needed from the user
            "respond_directly": "prepare_response", # If Gemini can respond directly
            "error": "prepare_response"         # If an error occurred in agent_node
        },
    )

    # After a tool is executed, move to prepare the final response
    workflow.add_edge("tool_executor", "prepare_response")

    # After asking for info, the flow goes back to the agent to process the user's next input
    workflow.add_edge("ask_for_info", "agent")

    # The final response from 'prepare_response' marks the end of a turn
    workflow.add_edge("prepare_response", END)

    return workflow.compile()

# --- Example Usage (for local testing and demonstration) ---
if __name__ == "__main__":
    # Setup the database for local testing
    Base.metadata.create_all(bind=engine)

    # Add some dummy data to the database for testing purposes
    session = next(get_db())

    # Clear existing data for clean tests
    session.query(Booking).delete()
    session.query(AvailabilitySlot).delete()
    session.query(Restaurant).delete()
    session.query(Customer).delete()
    session.query(CancellationReason).delete()
    session.commit()

    # Add a restaurant
    restaurant1 = Restaurant(name="The Fancy Fork", address="123 Main St", phone_number="555-1234")
    session.add(restaurant1)
    session.commit()
    session.refresh(restaurant1)

    # Add availability slots for the restaurant
    session.add_all([
        AvailabilitySlot(restaurant_id=restaurant1.id, date=date(2025, 12, 25), time=time(18, 0, 0), max_party_size=10, available=True),
        AvailabilitySlot(restaurant_id=restaurant1.id, date=date(2025, 12, 25), time=time(19, 0, 0), max_party_size=8, available=True),
        AvailabilitySlot(restaurant_id=restaurant1.id, date=date(2025, 12, 25), time=time(20, 0, 0), max_party_size=6, available=True),
        AvailabilitySlot(restaurant_id=restaurant1.id, date=date(2025, 12, 26), time=time(18, 0, 0), max_party_size=12, available=True),
        AvailabilitySlot(restaurant_id=restaurant1.id, date=date(2025, 12, 26), time=time(19, 0, 0), max_party_size=12, available=True),
    ])
    session.commit()

    # Add cancellation reasons
    session.add_all([
        CancellationReason(id=1, reason="Change of plans", description="Customer changed their mind."),
        CancellationReason(id=2, reason="Restaurant closed", description="Restaurant closed unexpectedly."),
        CancellationReason(id=3, reason="Duplicate booking", description="Accidental duplicate booking."),
    ])
    session.commit()

    # Add a customer
    customer1 = Customer(
        first_name="John", surname="Doe", email="john.doe@example.com", mobile="1234567890",
        receive_email_marketing=True
    )
    session.add(customer1)
    session.commit()
    session.refresh(customer1)

    # Add an existing booking for John Doe
    existing_booking = Booking(
        booking_reference="ABCDEFG",
        restaurant_id=restaurant1.id,
        customer_id=customer1.id,
        visit_date=date(2025, 12, 26),
        visit_time=time(19, 0, 0),
        party_size=4,
        channel_code="ONLINE",
        status="confirmed"
    )
    session.add(existing_booking)
    session.commit()
    session.close()

    # Compile the graph
    app_graph = create_booking_agent_graph()

    def run_conversation(graph, initial_message: str, thread_id: str = "test_thread_1"):
        """Helper function to run a single turn or multi-turn conversation."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initialize state for the first turn
        initial_state = AgentState(
            user_message=initial_message,
            chat_history=[],
            current_intent=None, tool_output=None, missing_info=[], response=None, original_tool_intent=None,
            restaurant_name=None, VisitDate=None, VisitTime=None, PartySize=None, ChannelCode=None,
            SpecialRequests=None, IsLeaveTimeConfirmed=None, RoomNumber=None, booking_reference=None,
            micrositeName=None, cancellationReasonId=None, customer_email=None, customer_first_name=None,
            customer_surname=None, customer_mobile=None, customer_title=None,
            customer_mobile_country_code=None, customer_phone_country_code=None, customer_phone=None,
            customer_receive_email_marketing=None, customer_receive_sms_marketing=None,
            customer_group_email_marketing_opt_in_text=None, customer_group_sms_marketing_opt_in_text=None,
            customer_receive_restaurant_email_marketing=None, customer_receive_restaurant_sms_marketing=None,
            customer_restaurant_email_marketing_opt_in_text=None, customer_restaurant_sms_marketing_opt_in_text=None,
        )

        current_state = initial_state
        print(f"User: {initial_message}")

        while True:
            for s in graph.stream(current_state, config):
                logger.info(f"Stream update: {s}")
                if "__end__" in s:
                    final_state = s["__end__"]
                    print(f"AI: {final_state['response']}")
                    
                    # Update chat history for next turn
                    current_state['chat_history'].append({"role": "user", "content": current_state['user_message']})
                    current_state['chat_history'].append({"role": "assistant", "content": final_state['response']})
                    
                    # If it was a multi-turn, keep the state for the next user input
                    if final_state.get('current_intent') == "ask_for_info":
                        next_message = input("You: ")
                        current_state['user_message'] = next_message
                        current_state['missing_info'] = [] # Reset missing info for new turn processing
                        current_state['response'] = None
                        current_state['tool_output'] = None
                        # Values populated in current_state via agent_node will persist,
                        # new values from next_message will overwrite or add.
                    else:
                        return # End of conversation turn for this example
            break # Exit the loop if __end__ is not found (shouldn't happen for complete turns)

    print("--- Conversation 1: Check Availability (Multi-turn Example) ---")
    run_conversation(app_graph, "Is The Fancy Fork available on 2025-12-25?", thread_id="avail_thread_1")

    print("\n--- Conversation 2: Create Booking (Direct Example) ---")
    run_conversation(app_graph, "Book a table at The Fancy Fork for 2 on 2025-12-26 at 18:00:00. My name is Jane Doe, email jane.doe@example.com, mobile 0987654321.", thread_id="book_thread_1")

    print("\n--- Conversation 3: Get Booking Details Example ---")
    run_conversation(app_graph, "What's the status of my booking ABCDEFG at The Fancy Fork?", thread_id="get_book_thread_1")

    print("\n--- Conversation 4: List Cancellation Reasons Example ---")
    run_conversation(app_graph, "What are the reasons I can choose for cancelling a booking?", thread_id="cancel_reasons_thread_1")

    print("\n--- Conversation 5: Cancel Booking (Multi-turn Example) ---")
    run_conversation(app_graph, "I want to cancel my booking ABCDEFG at The Fancy Fork.", thread_id="cancel_thread_1")
    # In the console, when prompted for reason ID, enter '1' for "Change of plans"

    print("\n--- Conversation 6: Update Booking Example ---")
    run_conversation(app_graph, "Can I change my booking ABCDEFG at The Fancy Fork to 3 people?", thread_id="update_thread_1")

    print("\n--- Conversation 7: Customer Bookings Summary Example ---")
    run_conversation(app_graph, "What bookings do I have? My email is john.doe@example.com.", thread_id="customer_bookings_thread_1")

    print("\n--- Conversation 8: Get Restaurants (Specific) Example ---")
    run_conversation(app_graph, "Are there any restaurants called 'The Fancy Fork'?", thread_id="get_restaurant_specific_thread_1")

    print("\n--- Conversation 9: Get All Restaurants Example ---")
    run_conversation(app_graph, "List all restaurants.", thread_id="get_all_restaurants_thread_1")

    # Clean up dummy data after tests
    session = next(get_db())
    session.query(Booking).delete()
    session.query(AvailabilitySlot).delete()
    session.query(Restaurant).delete()
    session.query(Customer).delete()
    session.query(CancellationReason).delete()
    session.commit()
    session.close()
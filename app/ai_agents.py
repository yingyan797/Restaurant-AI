import os, sys, json
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from datetime import datetime, date, time
import re # Import the regex module for stripping markdown

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAnKWcDA8jjl_Rgf1gJcdm_UBNB2UzWHXo"

# Add the directory containing ai_tools.py to the Python path
# Assuming ai_tools.py is in the same directory as this script.
# In a production environment, ensure your module paths are correctly configured.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import AIToolCallingInterface and other necessary components from your application modules.
try:
    from app.ai_tools import AIToolCallingInterface, logger
    # MOCK_BEARER_TOKEN and get_db are internally handled by AIToolCallingInterface methods.
except ImportError as e:
    print(f"Error: Could not import ai_tools module. Please ensure 'ai_tools.py' and its "
          f"dependencies (like 'app.database', 'app.models', 'app.routers.availability', "
          f"'app.routers.booking') are correctly placed and accessible in your Python path.")
    print(f"Details: {e}")
    sys.exit(1) # Exit if essential module cannot be imported


# --- LangChain Tool Definitions ---
from langchain.tools import tool

@tool
def search_availability_tool(restaurant_name: str, visit_date: str, party_size: int, channel_code: str = "ONLINE") -> Dict[str, Any]:
    """Search for available slots at a restaurant.
    Args:
        restaurant_name (str): The name of the restaurant.
        visit_date (str): The date of the visit in YYYY-MM-DD format (e.g., "2025-08-15").
        party_size (int): The number of people in the party.
        channel_code (str, optional): The channel code (e.g., "ONLINE"). Defaults to "ONLINE".
    """
    logger.info(f"Calling AIToolCallingInterface.search_availability with restaurant_name={restaurant_name}, VisitDate={visit_date}, PartySize={party_size}")
    return AIToolCallingInterface.search_availability(restaurant_name, visit_date, party_size, channel_code)

@tool
def create_booking_tool(
    restaurant_name: str, visit_date: str, visit_time: str, party_size: int,
    customer_email: str, customer_first_name: str, customer_surname: str, customer_mobile: str,
    channel_code: str = "ONLINE", special_requests: Optional[str] = None,
    is_leave_time_confirmed: Optional[bool] = False, room_number: Optional[str] = None,
    customer_title: Optional[str] = None, customer_mobile_country_code: Optional[str] = None,
    customer_phone_country_code: Optional[str] = None, customer_phone: Optional[str] = None,
    customer_receive_email_marketing: Optional[bool] = False, customer_receive_sms_marketing: Optional[bool] = False,
    customer_group_email_marketing_opt_in_text: Optional[str] = None, customer_group_sms_marketing_opt_in_text: Optional[str] = None,
    customer_receive_restaurant_email_marketing: Optional[bool] = False, customer_receive_restaurant_sms_marketing: Optional[bool] = False,
    customer_restaurant_email_marketing_opt_in_text: Optional[str] = None, customer_restaurant_sms_marketing_opt_in_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new booking.
    Args:
        restaurant_name (str): The name of the restaurant.
        visit_date (str): The date of the visit in YYYY-MM-DD format.
        visit_time (str): The time of the visit in HH:MM:SS format.
        party_size (int): The number of people in the party.
        customer_email (str): The customer's email address.
        customer_first_name (str): The customer's first name.
        customer_surname (str): The customer's surname.
        customer_mobile (str): The customer's mobile phone number.
        ... (other optional customer and booking details)
    """
    logger.info(f"Calling AIToolCallingInterface.create_booking for {restaurant_name} on {visit_date} at {visit_time} for {party_size} people.")
    return AIToolCallingInterface.create_booking(
        restaurant_name=restaurant_name, VisitDate=visit_date, VisitTime=visit_time,
        PartySize=party_size, customer_email=customer_email, customer_first_name=customer_first_name,
        customer_surname=customer_surname, customer_mobile=customer_mobile, ChannelCode=channel_code,
        SpecialRequests=special_requests, IsLeaveTimeConfirmed=is_leave_time_confirmed, RoomNumber=room_number,
        customer_title=customer_title, customer_mobile_country_code=customer_mobile_country_code,
        customer_phone_country_code=customer_phone_country_code, customer_phone=customer_phone,
        customer_receive_email_marketing=customer_receive_email_marketing, customer_receive_sms_marketing=customer_receive_sms_marketing,
        customer_group_email_marketing_opt_in_text=customer_group_email_marketing_opt_in_text,
        customer_group_sms_marketing_opt_in_text=customer_group_sms_marketing_opt_in_text,
        customer_receive_restaurant_email_marketing=customer_receive_restaurant_email_marketing,
        customer_receive_restaurant_sms_marketing=customer_receive_restaurant_sms_marketing,
        customer_restaurant_email_marketing_opt_in_text=customer_restaurant_email_marketing_opt_in_text,
        customer_restaurant_sms_marketing_opt_in_text=customer_restaurant_sms_marketing_opt_in_text,
    )

@tool
def get_booking_details_tool(restaurant_name: str, booking_reference: str) -> Dict[str, Any]:
    """Get details for a specific booking.
    Args:
        restaurant_name (str): The name of the restaurant.
        booking_reference (str): The unique booking reference.
    """
    logger.info(f"Calling AIToolCallingInterface.get_booking_details for booking {booking_reference} at {restaurant_name}")
    return AIToolCallingInterface.get_booking_details(restaurant_name, booking_reference)

@tool
def update_booking_details_tool(
    restaurant_name: str, booking_reference: str,
    visit_date: Optional[str] = None, visit_time: Optional[str] = None, party_size: Optional[int] = None,
    special_requests: Optional[str] = None, is_leave_time_confirmed: Optional[bool] = None
) -> Dict[str, Any]:
    """Update details for a specific booking.
    Args:
        restaurant_name (str): The name of the restaurant.
        booking_reference (str): The unique booking reference.
        visit_date (str, optional): New date of the visit in YYYY-MM-DD format.
        visit_time (str, optional): New time of the visit in HH:MM:SS format.
        party_size (int, optional): New number of people.
        special_requests (str, optional): Updated special requests.
        is_leave_time_confirmed (bool, optional): Whether leave time is confirmed.
    """
    logger.info(f"Calling AIToolCallingInterface.update_booking_details for booking {booking_reference} at {restaurant_name}")
    return AIToolCallingInterface.update_booking_details(
        restaurant_name, booking_reference, VisitDate=visit_date, VisitTime=visit_time,
        PartySize=party_size, SpecialRequests=special_requests, IsLeaveTimeConfirmed=is_leave_time_confirmed
    )

@tool
def cancel_booking_tool(restaurant_name: str, booking_reference: str, microsite_name: str, cancellation_reason_id: int) -> Dict[str, Any]:
    """Cancel a booking.
    Args:
        restaurant_name (str): The name of the restaurant.
        booking_reference (str): The unique booking reference.
        microsite_name (str): The unique microsite name of the restaurant.
        cancellation_reason_id (int): The ID of the cancellation reason.
    """
    logger.info(f"Calling AIToolCallingInterface.cancel_booking for booking {booking_reference} at {restaurant_name} with reason ID {cancellation_reason_id}")
    return AIToolCallingInterface.cancel_booking(restaurant_name, booking_reference, microsite_name, cancellation_reason_id)

@tool
def get_customer_information_tool(email: str) -> Dict[str, Any]:
    """Get customer information by email.
    Args:
        email (str): The customer's email address.
    """
    logger.info(f"Calling AIToolCallingInterface.get_customer_information for email {email}")
    return AIToolCallingInterface.get_customer_information(email)

@tool
def customer_bookings_and_restaurants_summary_tool(email: str, sql_conditions: List[str]) -> List[Dict[str, Any]]:
    """Get a summary of customer bookings and associated restaurant names.
    Args:
        email (str): User's email address.
        sql_conditions (List[str]): List of SQL WHERE conditions for bookings table (e.g., ["party_size > 2", "visit_date >= '2024-12-25'"]).
                                    Do NOT include 'where' clause, just the conditions.
                                    If no conditions are needed, pass an empty list [].
    """
    logger.info(f"Calling AIToolCallingInterface.customer_bookings_and_restaurants_summary for email {email} with conditions {sql_conditions}")
    return AIToolCallingInterface.customer_bookings_and_restaurants_summary(email, sql_conditions)

@tool
def list_cancellation_reasons_tool() -> List[Dict[str, Any]]:
    """List all possible cancellation reasons with ID, title, and description.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a cancellation reason.
                              Example: [{"id": 1, "reason": "Customer Cancelled", "description": "Customer initiated cancellation."}]
    """
    logger.info("Calling AIToolCallingInterface.list_cancellation_reasons")
    return AIToolCallingInterface.list_cancellation_reasons()

@tool
def find_restaurants_tool(sql_condition: str) -> List[Dict[str, Any]]:
    """Get information of all restaurants satisfying certain constraints.
    Args:
        sql_condition (str): SQL WHERE clause for the restaurants table. Example: "WHERE name='The Italian Place'" or "WHERE cuisine='Italian'".
                             If no specific conditions, pass "" (empty string) to get all restaurants.
    """
    logger.info(f"Calling AIToolCallingInterface.find_restaurants with condition '{sql_condition}'")
    return AIToolCallingInterface.find_restaurants(sql_condition)

# List of all tools available to the agent
tools = [
    search_availability_tool, create_booking_tool, get_booking_details_tool,
    update_booking_details_tool, cancel_booking_tool, get_customer_information_tool,
    customer_bookings_and_restaurants_summary_tool, list_cancellation_reasons_tool,
    find_restaurants_tool
]

# Initialize LLM
model = init_chat_model("google_genai:gemini-2.0-flash-lite").bind_tools(tools)

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our agent in the LangGraph workflow.
    """
    messages: List[BaseMessage]
    current_tool_call: Optional[Dict[str, Any]] # Stores tool call info if LLM decides to call a tool or manually prepared
    extracted_params: Dict[str, Any] # Stores parameters extracted from conversation (e.g., restaurant_name, visit_date)
    pending_questions: List[str] # Questions to ask the user for missing information
    bookings_summary: List[Dict[str, Any]] # List of bookings retrieved for the current customer
    selected_booking_reference: Optional[str] # The booking reference selected by the user from bookings_summary
    cancellation_reasons: List[Dict[str, Any]] # List of available cancellation reasons
    requested_action: Optional[str] # The high-level action the user wants to perform (e.g., "make_booking", "cancel_booking", "list_customer_bookings", "check_booking", "find_restaurants")
    user_email: str # The user's email, persistent throughout the session
    microsite_name: Optional[str] # Restaurant's microsite name, needed for cancel booking and other operations

# --- Helper Functions and Nodes ---

# Define required parameters for each action type
REQUIRED_PARAMS = {
    "find_availability": ["restaurant_name", "visit_date", "party_size"],
    "make_booking": ["restaurant_name", "visit_date", "visit_time", "party_size", "customer_first_name", "customer_surname", "customer_mobile"],
    "list_customer_bookings": [], # SQL conditions are optional, email is always known.
    "check_booking": ["restaurant_name", "booking_reference"], # For specific booking lookup
    "update_booking": [], # Handled by selection/direct input, then check updates. Requires booking_reference & restaurant_name.
    "cancel_booking": [], # Handled by selection/direct input, then check reasons. Requires booking_reference, restaurant_name, microsite_name, cancellation_reason_id.
    "find_restaurants": [], # SQL condition is optional, LLM should default or ask.
}

def get_required_params_for_action(action: str) -> List[str]:
    """Returns the list of required parameters for a given action."""
    return REQUIRED_PARAMS.get(action, [])

def get_extracted_value(state: AgentState, param_name: str) -> Any:
    """Helper to get a parameter value from extracted_params or other state."""
    if param_name in state["extracted_params"]:
        return state["extracted_params"][param_name]
    if param_name == "user_email":
        return state["user_email"]
    if param_name == "selected_booking_reference":
        return state["selected_booking_reference"]
    if param_name == "microsite_name":
        return state["microsite_name"]
    return None

def _list_of_dicts_to_html_table(data: List[Dict[str, Any]]) -> str:
    """Converts a list of dictionaries into an HTML table string."""
    if not data:
        return "<p>No results found.</p>"

    # Extract headers from the keys of the first dictionary, ensure consistent order
    # Sort headers for consistent display (optional but good for UX)
    all_keys = set().union(*(d.keys() for d in data))
    headers = sorted(list(all_keys))

    html = '<table border="1">'
    html += '<thead><tr>'
    for header in headers:
        html += f'<th>{header.replace("_", " ").title()}</th>'
    html += '</tr></thead><tbody>'

    for row in data:
        html += '<tr>'
        for header in headers:
            value = row.get(header, '')
            # Basic serialization for complex types (like date/time objects)
            if isinstance(value, (date, time, datetime)):
                value = value.isoformat()
            html += f'<td>{value}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

def call_llm(state: AgentState) -> Dict[str, Any]:
    """Invokes the LLM to determine the next action (tool call or direct response)."""
    messages = state['messages']

    # If the last message is a ToolMessage, ensure it's included for context
    # The LLM's initial system prompt in `initial_agent_state` will guide it on
    # how to interpret ToolMessage content (especially HTML).
    response = model.invoke(messages)

    logger.info(f"LLM Response: {response}")

    if response.tool_calls:
        # Ensure current_tool_call always stores a single tool call dictionary
        # LangChain's response.tool_calls is a list, take the first one as per design assumption.
        tool_call_dict = response.tool_calls[0]
        return {"messages": messages + [response], "current_tool_call": tool_call_dict}
    else:
        # If no tool call, it's a direct response, clear current_tool_call
        return {"messages": messages + [response], "current_tool_call": None}


def extract_parameters(state: AgentState) -> Dict[str, Any]:
    """
    Extracts parameters from the latest user message and updates the state.
    Also identifies the requested action. This uses an LLM call for robust parsing.
    """
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        return state # Only process human messages

    # System prompt for LLM to extract intent and parameters
    # Updated to include instructions for SQL conditions and refined requested_action logic
    extraction_system_prompt = f"""
    You are an AI assistant for a restaurant booking system. Your task is to identify the user's intent and extract relevant parameters from their message.

    **Identify one of the following `requested_action` values based on user intent:**
    - `find_availability`: User wants to search for available slots.
    - `make_booking`: User wants to create a new booking.
    - `list_customer_bookings`: User wants to see a summary of their past or future bookings. (e.g., "show my bookings", "what are my reservations?")
    - `check_booking`: User wants to get specific details for a single booking, usually by providing a booking reference or specific restaurant and date. (e.g., "check booking XYZ", "what's the status of my booking at The Italian Place for tomorrow?")
    - `update_booking`: User wants to modify an existing booking.
    - `cancel_booking`: User wants to cancel an existing booking.
    - `find_restaurants`: User wants to find information about restaurants.
    - `none`: No clear action identified.

    **Extract SQL WHERE conditions:**
    - For `find_restaurants` tool, extract a single string named `sql_condition`. This should be a valid SQL WHERE clause for the `restaurants` table. Example: "WHERE name='The Italian Place' AND microsite_name='Italian'".
    - For `customer_bookings_and_restaurants_summary` (implied by `list_customer_bookings`), extract a list of strings named `sql_conditions`, each being a valid SQL consition for the `bookings` table or the `restaurants` table. Example: `["bookings.party_size > 2", "restaurant.name = 'Italian7'"]`. If no specific conditions, set `sql_conditions` to `[]`.
    - Allowed filter variables for `restaurants` table: 'name', 'microsite_name'
    - Allowed filter variables for `bookings` table: 'visit_date', 'visit_time', 'party_size', 'channel_code', 'status', 'channel_code', 'special_requests', 'is_leave_time_confirmed', 'room_number'
    - Only extract a condition if it can surely be read or inferred from the user's conversation.

    Here are the possible parameters for `extracted_params` (use snake_case):
    - `restaurant_name`: str (e.g., "The Italian Place")
    - `visit_date`: str (YYYY-MM-DD, e.g., "2025-08-15") - **Current Date: {datetime.now().strftime('%Y-%m-%d')}**
    - `visit_time`: str (HH:MM:SS or HH:MM, e.g., "19:00:00" or "19:00")
    - `party_size`: int (e.g., 4)
    - `booking_reference`: str (e.g., "XYZ123")
    - `customer_email`: str (e.g., "john.doe@example.com")
    - `customer_first_name`: str (e.g., "John")
    - `customer_surname`: str (e.g., "Doe")
    - `customer_mobile`: str (e.g., "07700900123")
    - `cancellation_reason_id`: int (refer to ListCancellationReasonsTool for IDs)
    - `special_requests`: str
    - `is_leave_time_confirmed`: bool (true/false)
    - `sql_condition`: str (for `find_restaurants` tool")
    - `sql_conditions`: list[str] (for `customer_bookings_and_restaurants_summary` tool)
    If the user provides a number (e.g., "1") after being asked to select a booking or cancellation reason, only set 'selected_booking_reference' or 'cancellation_reason_id' respectively in `extracted_params`. Do NOT set a `requested_action` for these numerical inputs.

    Provide the output as a JSON object with 'requested_action' and 'extracted_params' keys.
    Example:
    {{
      "requested_action": "find_availability",
      "extracted_params": {{
        "restaurant_name": "The Italian Place",
        "visit_date": "{ (datetime.now()).strftime('%Y-%m-%d') }",
        "party_size": 4
      }}
    }}

    """
    extraction_prompt = extraction_system_prompt + f"User message: {last_message.content}"

    try:
        response_parsing = model.invoke([HumanMessage(content=extraction_prompt)])
        raw_content = response_parsing.content.strip()

        # Regex to strip markdown code block fences if present
        # This will handle `json` or general ` ``` ` wrapping
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
        json_string = match.group(1) if match else raw_content

        parsed_output = json.loads(json_string) # Use the potentially stripped string
        new_extracted_params = parsed_output.get("extracted_params", {})
        requested_action_from_llm = parsed_output.get("requested_action")

        # Update state: prioritize new extracted params
        state['extracted_params'].update(new_extracted_params)
        if requested_action_from_llm:
            state['requested_action'] = requested_action_from_llm

        # Update user_email if provided in conversation
        if "customer_email" in new_extracted_params and new_extracted_params["customer_email"]:
            state["user_email"] = new_extracted_params["customer_email"]

        # If a restaurant name is extracted, try to get its microsite name for future use (e.g., cancellation)
        # This is a pre-emptive fetch, it should ideally happen only once per restaurant or when needed.
        if state['extracted_params'].get("restaurant_name") and not state.get("microsite_name"):
            restaurant_name = state['extracted_params']["restaurant_name"]
            try:
                db_session = AIToolCallingInterface.get_db_session()
                if db_session:
                    # Using a direct SQL condition for exact match
                    restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{restaurant_name}'")
                    if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                        state["microsite_name"] = restaurant_info[0].get("microsite_name")
                        logger.info(f"Retrieved microsite_name: {state['microsite_name']} for {restaurant_name}")
                    else:
                        logger.warning(f"No restaurant found or invalid response for name: {restaurant_name}")
                    db_session.close()
            except Exception as e:
                logger.error(f"Could not retrieve microsite_name for {restaurant_name}: {e}")

    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM extraction output (not valid JSON): {raw_content}") # Log raw content for debugging
        state["messages"].append(AIMessage(content="I had trouble understanding your request. Could you please rephrase it?"))
        state["requested_action"] = "none" # Reset action
    except Exception as e:
        logger.error(f"Error during parameter extraction: {e}")
        state["messages"].append(AIMessage(content=f"An internal error occurred during processing: {e}"))
        state["requested_action"] = "none" # Reset action

    return state

def check_missing_params(state: AgentState) -> Dict[str, Any]:
    """
    Checks for missing required parameters for the requested action.
    If parameters are missing, sets pending_questions.
    """
    requested_action = state.get("requested_action")
    if not requested_action or requested_action == "none":
        return state # No specific action identified

    required_params = get_required_params_for_action(requested_action)
    missing_params = []

    # Special handling for actions that derive required parameters from selection or existing state
    if requested_action in ["update_booking", "cancel_booking", "check_booking"]:
        # These primarily need selected_booking_reference and restaurant_name
        if not state.get("selected_booking_reference") and not state["extracted_params"].get("booking_reference"):
            missing_params.append("booking_reference")
        if not state["extracted_params"].get("restaurant_name"):
            missing_params.append("restaurant_name")
        # For cancel, also check microsite and cancellation reason
        if requested_action == "cancel_booking":
            if not state.get("microsite_name"):
                missing_params.append("microsite_name")
            if state["extracted_params"].get("cancellation_reason_id") is None:
                missing_params.append("cancellation_reason_id")
    else: # For other actions (find_availability, make_booking) check directly from extracted_params
        for param in required_params:
            if get_extracted_value(state, param) is None:
                missing_params.append(param)
    
    state["pending_questions"] = [] # Clear previous questions

    if missing_params:
        question_parts = [p.replace('_', ' ') for p in missing_params]
        if len(question_parts) == 1:
            state["pending_questions"].append(f"What is the {question_parts[0]}?")
        else:
            state["pending_questions"].append(f"I need the following information: {', '.join(question_parts)}.")
    return state


def ask_for_clarification(state: AgentState) -> Dict[str, Any]:
    """Generates a message asking the user for missing information."""
    questions = state.get("pending_questions", ["What else can I help you with?"])
    response_content = " ".join(questions)
    # Add the AI's question to messages, not just update state
    state["messages"].append(AIMessage(content=response_content))
    return state

def get_customer_bookings(state: AgentState) -> Dict[str, Any]:
    """Fetches customer bookings using AIToolCallingInterface."""
    user_email = state.get("user_email")
    if not user_email:
        state["pending_questions"] = ["I need your email address to retrieve your bookings."]
        return state # Routing will send to ask_for_clarification

    sql_conditions = state["extracted_params"].get("sql_conditions", [])

    try:
        bookings = AIToolCallingInterface.customer_bookings_and_restaurants_summary(user_email, sql_conditions)
        if isinstance(bookings, dict) and "error" in bookings:
            raise Exception(bookings["error"])

        state["bookings_summary"] = bookings
        logger.info(f"Fetched customer bookings: {bookings}")
    except Exception as e:
        logger.error(f"Error fetching customer bookings: {e}")
        state["messages"].append(AIMessage(content=f"An error occurred while fetching your bookings: {e}"))

    return state

def handle_multiple_bookings(state: AgentState) -> Dict[str, Any]:
    """Presents multiple bookings to the user and asks for selection."""
    bookings = state.get("bookings_summary", [])
    if not bookings:
        state["messages"].append(AIMessage(content="No bookings found to display for selection."))
        # Clear selected action as we can't proceed without a booking.
        state["requested_action"] = None
        state["pending_questions"] = []
        return state

    summary_messages = ["I found multiple bookings for you:"]
    # Pass the list of dicts to the HTML table formatter
    html_table = _list_of_dicts_to_html_table(bookings)
    summary_messages.append(html_table) # Add the HTML table
    
    summary_messages.append("Please tell me the number of the booking you'd like to proceed with (e.g., '1').")
    state["pending_questions"] = ["Please select a booking by its number."]
    state["messages"].append(AIMessage(content="\n".join(summary_messages)))
    return state

def process_selected_booking(state: AgentState) -> Dict[str, Any]:
    """
    Processes the user's selection for a booking from a list.
    Assumes the last message contains a number.
    """
    last_message_content = state['messages'][-1].content
    bookings = state.get("bookings_summary", [])

    try:
        selection_index = int(last_message_content) - 1
        if 0 <= selection_index < len(bookings):
            selected_booking = bookings[selection_index]
            state["selected_booking_reference"] = selected_booking["booking_reference"]
            state["extracted_params"]["restaurant_name"] = selected_booking["restaurant_name"]
            # Clear pending questions as selection is made
            state["pending_questions"] = []
            logger.info(f"User selected booking: {selected_booking['booking_reference']}")

            # Try to get microsite name for the selected restaurant if not already present
            if not state.get("microsite_name") and state["extracted_params"].get("restaurant_name"):
                restaurant_name = state["extracted_params"]["restaurant_name"]
                try:
                    db_session = AIToolCallingInterface.get_db_session()
                    if db_session:
                        restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{restaurant_name}'")
                        if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                            state["microsite_name"] = restaurant_info[0].get("microsite_name")
                            logger.info(f"Retrieved microsite_name: {state['microsite_name']} for {restaurant_name}")
                        db_session.close()
                except Exception as e:
                    logger.warning(f"Could not retrieve microsite_name for {restaurant_name}: {e}")

        else:
            response_content = "That's not a valid selection. Please choose a number from the list."
            state["messages"].append(AIMessage(content=response_content))
            state["pending_questions"] = ["Please select a valid booking number."] # Re-ask
    except ValueError:
        response_content = "I didn't understand your selection. Please provide the number of the booking."
        state["messages"].append(AIMessage(content=response_content))
        state["pending_questions"] = ["Please provide a number."] # Re-ask

    return state


def get_cancellation_reasons(state: AgentState) -> Dict[str, Any]:
    """Fetches cancellation reasons and asks user to choose."""
    try:
        reasons = AIToolCallingInterface.list_cancellation_reasons()
        if isinstance(reasons, dict) and "error" in reasons:
            raise Exception(reasons["error"])

        state["cancellation_reasons"] = reasons
        if reasons:
            reason_messages = ["Please select a cancellation reason by its ID:"]
            # Convert reasons to a list of dicts suitable for HTML table display
            reasons_for_display = [{"ID": r['id'], "Reason": r['reason'], "Description": r['description']} for r in reasons]
            reason_messages.append(_list_of_dicts_to_html_table(reasons_for_display))

            state["pending_questions"] = ["Please provide the ID of the cancellation reason."]
            state["messages"].append(AIMessage(content="\n".join(reason_messages)))
        else:
            state["messages"].append(AIMessage(content="Could not retrieve cancellation reasons. Please try again."))
    except Exception as e:
        logger.error(f"Error fetching cancellation reasons: {e}")
        state["messages"].append(AIMessage(content=f"An error occurred while fetching cancellation reasons: {e}"))
    return state


def call_tool(state: AgentState) -> Dict[str, Any]:
    """Executes the identified tool call stored in state."""
    tool_call_info = state.get("current_tool_call")
    if not tool_call_info:
        logger.warning("call_tool node called without a current_tool_call. This should not happen if routing is correct.")
        return {"messages": state["messages"] + [AIMessage(content="Internal error: No tool call information found.")]}

    tool_name = tool_call_info["name"]
    tool_args = tool_call_info["args"]
    tool_id = tool_call_info.get("id", f"auto_generated_id_{datetime.now().timestamp()}") # Get ID or use a dummy

    # Find the LangChain tool object by name
    selected_tool = next((t for t in tools if t.name == tool_name), None)

    if not selected_tool:
        logger.error(f"Attempted to call unknown tool: {tool_name}")
        return {"messages": state["messages"] + [AIMessage(content=f"Error: Attempted to call an unknown function: {tool_name}.")]}

    try:
        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
        tool_output = selected_tool.invoke(tool_args)
        logger.info(f"Raw Tool output for {tool_name}: {tool_output}")

        # Special handling for tools that return structured data for HTML rendering
        display_output = tool_output
        if isinstance(tool_output, list) and all(isinstance(item, dict) for item in tool_output):
            if tool_name in [search_availability_tool.name, customer_bookings_and_restaurants_summary_tool.name, find_restaurants_tool.name]:
                display_output = _list_of_dicts_to_html_table(tool_output)
                logger.info(f"Converted {tool_name} output to HTML.")
            
        # Convert output to JSON string for ToolMessage content if it's a dict or list
        if isinstance(display_output, (dict, list)):
            tool_content = json.dumps(display_output)
        else:
            tool_content = str(display_output) # Fallback for other types (e.g., direct HTML string)


        return {"messages": state["messages"] + [ToolMessage(content=tool_content, tool_call_id=tool_id)]}
    except Exception as e:
        logger.error(f"Error calling tool {tool_name} with args {tool_args}: {e}")
        error_message = f"An error occurred while executing {tool_name}: {e}"
        # Return a ToolMessage with error content so LLM can be aware
        return {"messages": state["messages"] + [ToolMessage(content=error_message, tool_call_id=tool_id)]}


def perform_action(state: AgentState) -> Dict[str, Any]:
    """
    This node serves as a dispatcher to prepare arguments and set `current_tool_call`
    for the appropriate tool based on the `requested_action`.
    """
    requested_action = state.get("requested_action")
    extracted_params = state.get("extracted_params", {})
    user_email = state.get("user_email")
    selected_booking_reference = state.get("selected_booking_reference")
    microsite_name = state.get("microsite_name") # Crucial for cancel

    args_for_tool = {}
    tool_to_call_name = None

    if requested_action == "find_availability":
        tool_to_call_name = search_availability_tool.name
        args_for_tool = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "visit_date": extracted_params.get("visit_date"),
            "party_size": extracted_params.get("party_size"),
            "channel_code": extracted_params.get("channel_code", "ONLINE")
        }

    elif requested_action == "make_booking":
        tool_to_call_name = create_booking_tool.name
        all_create_booking_args = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "visit_date": extracted_params.get("visit_date"),
            "visit_time": extracted_params.get("visit_time"),
            "party_size": extracted_params.get("party_size"),
            "customer_email": user_email, # Always use the state's user_email for booking
            "customer_first_name": extracted_params.get("customer_first_name"),
            "customer_surname": extracted_params.get("customer_surname"),
            "customer_mobile": extracted_params.get("customer_mobile"),
            "channel_code": extracted_params.get("channel_code", "ONLINE"),
            "special_requests": extracted_params.get("special_requests"),
            "is_leave_time_confirmed": extracted_params.get("is_leave_time_confirmed"),
            "room_number": extracted_params.get("room_number"),
            "customer_title": extracted_params.get("customer_title"),
            "customer_mobile_country_code": extracted_params.get("customer_mobile_country_code"),
            "customer_phone_country_code": extracted_params.get("customer_phone_country_code"),
            "customer_phone": extracted_params.get("customer_phone"),
            "customer_receive_email_marketing": extracted_params.get("customer_receive_email_marketing"),
            "customer_receive_sms_marketing": extracted_params.get("customer_receive_sms_marketing"),
            "customer_group_email_marketing_opt_in_text": extracted_params.get("customer_group_email_marketing_opt_in_text"),
            "customer_group_sms_marketing_opt_in_text": extracted_params.get("customer_group_sms_marketing_opt_in_text"),
            "customer_receive_restaurant_email_marketing": extracted_params.get("customer_receive_restaurant_email_marketing"),
            "customer_receive_restaurant_sms_marketing": extracted_params.get("customer_receive_restaurant_sms_marketing"),
            "customer_restaurant_email_marketing_opt_in_text": extracted_params.get("customer_restaurant_email_marketing_opt_in_text"),
            "customer_restaurant_sms_marketing_opt_in_text": extracted_params.get("customer_restaurant_sms_marketing_opt_in_text"),
        }
        args_for_tool = {k: v for k, v in all_create_booking_args.items() if v is not None} # Filter None

    elif requested_action == "list_customer_bookings":
        tool_to_call_name = customer_bookings_and_restaurants_summary_tool.name
        args_for_tool = {
            "email": user_email,
            "sql_conditions": extracted_params.get("sql_conditions", [])
        }

    elif requested_action == "check_booking":
        tool_to_call_name = get_booking_details_tool.name
        args_for_tool = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "booking_reference": extracted_params.get("booking_reference") or selected_booking_reference # Use specific if available, otherwise selected
        }
        if not args_for_tool["restaurant_name"] or not args_for_tool["booking_reference"]:
            state["messages"].append(AIMessage(content="To check a specific booking, I need both the restaurant name and the booking reference."))
            state["requested_action"] = None
            return state

    elif requested_action == "update_booking":
        tool_to_call_name = update_booking_details_tool.name
        args_for_tool = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "booking_reference": selected_booking_reference if selected_booking_reference else extracted_params.get("booking_reference")
        }
        update_params = ["visit_date", "visit_time", "party_size", "special_requests", "is_leave_time_confirmed"]
        for param in update_params:
            if extracted_params.get(param) is not None:
                args_for_tool[param] = extracted_params[param]
        if not args_for_tool["restaurant_name"] or not args_for_tool["booking_reference"]:
            state["messages"].append(AIMessage(content="To update a booking, I need the restaurant name and the booking reference."))
            state["requested_action"] = None
            return state

    elif requested_action == "cancel_booking":
        tool_to_call_name = cancel_booking_tool.name
        args_for_tool = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "booking_reference": selected_booking_reference if selected_booking_reference else extracted_params.get("booking_reference"),
            "microsite_name": microsite_name,
            "cancellation_reason_id": extracted_params.get("cancellation_reason_id")
        }
        if not args_for_tool["restaurant_name"] or not args_for_tool["booking_reference"] or not args_for_tool["microsite_name"] or args_for_tool["cancellation_reason_id"] is None:
            state["messages"].append(AIMessage(content="To cancel a booking, I need the restaurant name, booking reference, microsite name, and a cancellation reason ID."))
            state["requested_action"] = None
            return state

    elif requested_action == "find_restaurants":
        tool_to_call_name = find_restaurants_tool.name
        # Changed default sql_condition to an empty string as requested
        args_for_tool = {
            "sql_condition": extracted_params.get("sql_condition", "") # Default to empty string
        }
        
    else:
        state["messages"].append(AIMessage(content="I'm not sure how to perform that action based on the information provided."))
        state["requested_action"] = None # Clear action if unable to map
        return state

    if tool_to_call_name:
        # Manually create a tool call object to pass to the call_tool node
        mock_tool_call = {
            "name": tool_to_call_name,
            "args": args_for_tool,
            "id": f"manual_call_{tool_to_call_name}_{datetime.now().timestamp()}" # Unique ID
        }
        state["current_tool_call"] = mock_tool_call
        logger.info(f"Prepared tool call: {mock_tool_call}")

    return state # Router will pick up current_tool_call and route to call_tool


def decide_next_step(state: AgentState) -> str:
    """
    Main router function for the graph, determining the next node based on the current state.
    This function implements the core logic for multi-turn conversation and tool orchestration.
    """
    last_message = state['messages'][-1]
    requested_action = state.get("requested_action")
    extracted_params = state.get("extracted_params", {})
    pending_questions = state.get("pending_questions")
    bookings_summary = state.get("bookings_summary")
    selected_booking_reference = state.get("selected_booking_reference")
    cancellation_reasons = state.get("cancellation_reasons")
    current_tool_call = state.get("current_tool_call")

    logger.info(f"Deciding next step. Current state: requested_action={requested_action}, "
                f"pending_questions={pending_questions}, current_tool_call={current_tool_call is not None}, "
                f"bookings_summary_len={len(bookings_summary) if bookings_summary else 0}, "
                f"selected_booking_ref={selected_booking_reference}")

    # 1. If the last message was a tool message, it means a tool just ran.
    #    Now, feed the tool output back to the LLM to get a human-readable response.
    if isinstance(last_message, ToolMessage):
        # Clear current_tool_call after execution to prevent re-execution
        state["current_tool_call"] = None
        # Reset specific state variables for next interaction if the action is considered "complete"
        # or if LLM needs to summarize a listing.
        if requested_action in ["make_booking", "cancel_booking", "update_booking", "check_booking", "list_customer_bookings", "find_availability", "find_restaurants"]:
            # If the tool successfully handled the request, we reset the action to allow new queries.
            # However, if it was a listing (list_customer_bookings, find_restaurants, search_availability),
            # the LLM should summarize the results and then wait for next query.
            # The tool output has been added to messages, so now LLM will process it.
            pass # Continue to call_llm
        return "call_llm" # LLM will summarize tool output

    # 2. If there are pending questions, it means we just asked for info.
    #    Now we expect the user to have provided it, so route to process that input.
    if pending_questions:
        if "Please select a booking by its number." in pending_questions:
            return "process_selected_booking"
        elif "Please provide the ID of the cancellation reason." in pending_questions:
            # User provided a number for cancellation ID. Try to parse it directly.
            try:
                cancellation_id = int(last_message.content)
                state["extracted_params"]["cancellation_reason_id"] = cancellation_id
                state["pending_questions"] = [] # Clear question
                return "check_missing_params" # Recheck if all parameters for cancel are now present
            except ValueError:
                state["messages"].append(AIMessage(content="That's not a valid ID. Please provide a numeric ID for the cancellation reason."))
                return "get_cancellation_reasons" # Re-ask if invalid input for ID
        else:
            # General missing params, go back to extraction to capture user's answer
            return "extract_parameters"

    # 3. If LLM has decided to call a tool (and it's not a manual call from `perform_action`)
    # This path is taken directly after `call_llm` if it suggests a tool.
    if current_tool_call:
        return "call_tool"

    # 4. Handle specific action flows based on `requested_action`
    if requested_action:
        # Flow for listing and selecting bookings (for check, update, cancel, or just listing)
        if requested_action in ["list_customer_bookings", "check_booking", "update_booking", "cancel_booking"]:
            # If a booking hasn't been selected/identified yet
            if not selected_booking_reference:
                # If we haven't fetched bookings yet, do so
                if not bookings_summary:
                    return "get_customer_bookings"
                # If we have bookings, but more than one, ask user to select
                elif len(bookings_summary) > 1:
                    return "handle_multiple_bookings"
                # If only one booking, auto-select it and proceed
                elif len(bookings_summary) == 1:
                    state["selected_booking_reference"] = bookings_summary[0]["booking_reference"]
                    state["extracted_params"]["restaurant_name"] = bookings_summary[0]["restaurant_name"]
                    # If this is for cancellation, also try to get microsite name
                    if requested_action == "cancel_booking" and not state.get("microsite_name"):
                         try:
                            db_session = AIToolCallingInterface.get_db_session()
                            if db_session:
                                restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{state['extracted_params']['restaurant_name']}'")
                                if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                                    state["microsite_name"] = restaurant_info[0].get("microsite_name")
                                db_session.close()
                         except Exception as e:
                            logger.warning(f"Could not retrieve microsite_name for {state['extracted_params']['restaurant_name']}: {e}")
                    # After auto-selection, re-check params for the specific action (check/update/cancel)
                    return "check_missing_params"
                else: # No bookings found after get_customer_bookings
                    state["messages"].append(AIMessage(content="I couldn't find any bookings matching that criteria for your email."))
                    state["requested_action"] = None # Clear action
                    state["bookings_summary"] = [] # Clear summary
                    return "call_llm" # Let LLM provide general response

        # Cancellation specific: need cancellation reason ID
        if requested_action == "cancel_booking" and state["extracted_params"].get("cancellation_reason_id") is None:
            if not cancellation_reasons: # If reasons not fetched yet
                return "get_cancellation_reasons"
            else: # Reasons fetched, waiting for ID from user (will be caught by pending_questions)
                state["pending_questions"] = ["Please provide the ID of the cancellation reason."]
                return "ask_for_clarification"

        # If we have a requested action and no specific sub-flow (like selection or reasons) initiated,
        # then check if all parameters for that action are present.
        return "check_missing_params"

    # 5. Default: If no specific action, and no tool call from LLM yet,
    #    this is likely a new turn or a general query. Let LLM decide.
    return "call_llm"


# --- LangGraph Setup ---
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("call_llm", call_llm)
workflow.add_node("extract_parameters", extract_parameters)
workflow.add_node("check_missing_params", check_missing_params)
workflow.add_node("ask_for_clarification", ask_for_clarification)
workflow.add_node("get_customer_bookings", get_customer_bookings)
workflow.add_node("handle_multiple_bookings", handle_multiple_bookings)
workflow.add_node("process_selected_booking", process_selected_booking)
workflow.add_node("get_cancellation_reasons", get_cancellation_reasons)
workflow.add_node("call_tool", call_tool)
workflow.add_node("perform_action", perform_action) # Prepares tool call, then current_tool_call is picked up by router

# Define edges
# Entry point: Always start by extracting parameters from the user's message
workflow.set_entry_point("extract_parameters")

# After extracting parameters, decide the next step using the main router
# This is handled by the conditional edge above

# After asking for clarification, the next user message will be processed by extract_parameters
workflow.add_edge("ask_for_clarification", "extract_parameters")

# After getting customer bookings, route based on the number of bookings found
workflow.add_conditional_edges(
    "get_customer_bookings",
    decide_next_step,
    {
        "call_llm": "call_llm",
        "handle_multiple_bookings": "handle_multiple_bookings",
        "check_missing_params": "check_missing_params",
        END: END
    }
)

# After handling multiple bookings, the next user input is expected to be a selection, so back to extract_parameters
workflow.add_edge("handle_multiple_bookings", "extract_parameters")

# After processing selected booking, re-check parameters to proceed with the action
workflow.add_conditional_edges(
    "process_selected_booking",
    decide_next_step,
    {
        "check_missing_params": "check_missing_params",
        "ask_for_clarification": "ask_for_clarification",
        END: END
    }
)

# After getting cancellation reasons, the next user input is expected to be an ID, so back to extract_parameters
workflow.add_edge("get_cancellation_reasons", "extract_parameters")

# After `perform_action` prepares the tool call by setting `current_tool_call`, it then transitions to `call_tool`.
workflow.add_edge("perform_action", "call_tool")

# After `call_tool` executes, feed the tool output back to LLM for response generation
workflow.add_edge("call_tool", "call_llm")

# Main conditional routing logic
# This edge determines the flow after `extract_parameters`, `get_customer_bookings`, and other nodes
workflow.add_conditional_edges(
    "extract_parameters",
    decide_next_step, # The router function
    {
        "call_llm": "call_llm",
        "extract_parameters": "extract_parameters",
        "check_missing_params": "check_missing_params",
        "ask_for_clarification": "ask_for_clarification",
        "get_customer_bookings": "get_customer_bookings",
        "handle_multiple_bookings": "handle_multiple_bookings",
        "process_selected_booking": "process_selected_booking",
        "get_cancellation_reasons": "get_cancellation_reasons",
        "perform_action": "perform_action",
        "call_tool": "call_tool", # If current_tool_call is already set by LLM or manual routing
        END: END # End the conversation if LLM has a direct response or the task is complete
    }
)

# After `check_missing_params`, if no questions are pending, proceed to `perform_action`.
# Otherwise, go to `ask_for_clarification`.
workflow.add_conditional_edges(
    "check_missing_params",
    lambda state: "ask_for_clarification" if state["pending_questions"] else "perform_action",
    {
        "ask_for_clarification": "ask_for_clarification",
        "perform_action": "perform_action"
    }
)

# After `call_llm`, if it generated a tool call, route to `call_tool`. Otherwise, it's the end of the turn.
workflow.add_conditional_edges(
    "call_llm",
    lambda state: "call_tool" if state.get("current_tool_call") else END,
    {
        "call_tool": "call_tool",
        END: END
    }
)

# Compile the graph
graph = workflow.compile()

# Initial state for the agent
# The user's email is given for the entire session as per the prompt.
initial_agent_state = AgentState(
    messages=[
        # Initial system message for the LLM. This guides its behavior throughout the conversation.
        HumanMessage(content="""You are an AI assistant for restaurant bookings.
        When you receive tool output that is a JSON string representing an HTML table (starting with '{"table": "<table'), or a direct HTML string (starting with '<table' or '<p>'), your response **must be solely** that HTML content directly to the user. Do not add any introductory or concluding text, and **do not make any new tool calls in that turn.** Simply pass the HTML through as your response for displaying in a UI.
        For other tool outputs (e.g., success messages, single booking details, cancellation reasons list before selection), summarize the results in a user-friendly way.
        If you need to ask for information, make your questions clear and concise.
        Prioritize gathering missing information using the available tools and by asking the user.
        Be polite and helpful.
        """),
    ],
    current_tool_call=None,
    extracted_params={},
    pending_questions=[],
    bookings_summary=[],
    selected_booking_reference=None,
    cancellation_reasons=[],
    requested_action=None,
    user_email="yingyan797@restaurantai.com", # Example fixed email for the session
    microsite_name=None,
)

# --- Interactive Chat Loop ---
def run_chat_loop(initial_state: AgentState):
    """
    Runs an interactive chat loop with the LangGraph agent.
    """
    print("Welcome to the Restaurant Booking Assistant!")
    print(f"Your email for this session is: {initial_state['user_email']}")
    print("Type 'exit' to end the conversation.")
    print("Note: Structured data like booking lists and availability will be rendered as HTML tables.")
    print("\nHow can I help you today?")

    # Create a deep copy of the initial state for each session to avoid state leakage
    import copy
    current_state = copy.deepcopy(initial_state)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Append user message to the state
        current_state['messages'].append(HumanMessage(content=user_input))

        # Clear any previous current_tool_call or pending_questions before processing new input
        current_state["current_tool_call"] = None
        current_state["pending_questions"] = []
        # Clear selected_booking_reference and bookings_summary if the user starts a new main action
        # This is a heuristic to prevent stale data from affecting new queries.
        if current_state["requested_action"] is None: # Only reset if no action is currently pending
            current_state["bookings_summary"] = []
            current_state["selected_booking_reference"] = None
            current_state["cancellation_reasons"] = []
            current_state["microsite_name"] = None
            current_state["extracted_params"] = {} # Also clear extracted params for a fresh start


        try:
            # Initialize a temporary state to hold the latest valid state from stream
            # This ensures 'current_state' has the correct structure even if the stream is empty or fails.
            temp_state = copy.deepcopy(current_state) 

            # Stream through the graph execution steps
            for s in graph.stream(current_state):
                temp_state = s # Update temp_state with each yielded state
                # Uncomment for detailed debugging:
                # print(f"--- State after step: {temp_state} ---")

            current_state = temp_state # After loop, current_state is the last valid state from the stream

            # After the graph finishes a turn, print the AI's final response for this turn
            # Defensive check for 'messages' key and non-empty list
            if 'messages' not in current_state or not current_state['messages']:
                logger.error(f"Final state for chat loop is missing 'messages' key or it's empty. State: {current_state}")
                print("AI: I'm sorry, I encountered an internal issue and couldn't generate a response. The conversation has been reset. Please try again.")
                # Reset the conversation state to prevent cascading errors
                current_state = copy.deepcopy(initial_agent_state)
                # Preserve user email if it was already set (assuming initial_agent_state has user_email)
                current_state['user_email'] = initial_state['user_email'] 
                continue # Skip to next loop iteration

            final_message = current_state['messages'][-1]
            if isinstance(final_message, AIMessage):
                print(f"AI: {final_message.content}")
            elif isinstance(final_message, ToolMessage):
                # This case should ideally be rare for direct user output, as LLM typically interprets tool output.
                # However, if the LLM is instructed to pass HTML directly, it might not wrap it in an AIMessage.
                try:
                    # Check if the content is JSON that might contain HTML
                    content_obj = json.loads(final_message.content)
                    if isinstance(content_obj, str) and (content_obj.strip().startswith("<table") or content_obj.strip().startswith("<p>")):
                         print(f"AI (HTML Response): {content_obj}")
                    else:
                         print(f"AI (Tool Output - raw): {final_message.content}")
                except json.JSONDecodeError:
                    if final_message.content.strip().startswith("<table") or final_message.content.strip().startswith("<p>"):
                         print(f"AI (HTML Response): {final_message.content}")
                    else:
                        print(f"AI (Tool Output - raw): {final_message.content}")
            else:
                # This could happen if a node updates state but doesn't add an AIMessage
                print(f"AI: (Processing... please wait for next message or provide more info.)")

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print(f"AI: An unexpected error occurred: {e}. Please try again or rephrase your request.")
            # Reset the state to a known good point or initial state upon error
            current_state = copy.deepcopy(initial_agent_state)
            # Preserve user email if it was already set
            current_state['user_email'] = initial_state['user_email'] 
            current_state['messages'].append(AIMessage(content="I'm sorry, I encountered an error. Let's start fresh. How can I help you?"))


if __name__ == "__main__":
    # You can visualize the graph uncommenting the line below (requires graphviz installed)
    # graph.get_graph().draw_png("restaurant_booking_agent.png") # or .draw_ascii() for console

    run_chat_loop(initial_agent_state)
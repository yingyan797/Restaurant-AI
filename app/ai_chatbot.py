import os, sys, json
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from datetime import datetime, date, time
import re # Import the regex module for stripping markdown
import copy

# Set up Google API key - Replace with your actual key or secure loading
os.environ["GOOGLE_API_KEY"] = "AIzaSyAnKWcDA8jjl_Rgf1gJcdm_UBNB2UzWHXo"

# Add the directory containing ai_tools.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import AIToolCallingInterface and other necessary components
try:
    from app.ai_tools import AIToolCallingInterface, logger
except ImportError as e:
    print(f"Error: Could not import ai_tools module. Please ensure 'ai_tools.py' and its "
          f"dependencies are correctly placed and accessible in your Python path.")
    print(f"Details: {e}")
    sys.exit(1)


# --- LangChain Tool Definitions (Unchanged) ---
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
        sql_condition (str): SQL WHERE clause for the restaurants table. Example: "WHERE name='The Italian Place'".
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

# --- Agent State Definition (Unchanged) ---
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

# Define required parameters for each action type (Unchanged)
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

def call_llm(state: AgentState) -> AgentState:
    """Invokes the LLM to determine the next action (tool call or direct response)."""
    messages = state['messages']

    response = model.invoke(messages) # This model is bound with tools

    logger.info(f"LLM Response in call_llm: {response}")

    # Create new state by copying existing state
    new_state = {**state}
    new_state["messages"] = messages + [response]
    
    if response.tool_calls:
        # LLM decided to call a tool
        tool_call_dict = response.tool_calls[0]
        new_state["current_tool_call"] = tool_call_dict
    else:
        # LLM decided to respond directly without a tool call
        new_state["current_tool_call"] = None

    return new_state


def llm_summarize_tool_output(state: AgentState) -> AgentState:
    messages = state['messages'] # Get the full message history

    # The last message is guaranteed to be the ToolMessage
    last_tool_message = messages[-1]

    # Find the most recent HumanMessage in the history.
    last_human_message = None
    for i in range(len(messages) - 2, -1, -1): # Iterate backwards from before the ToolMessage
        if isinstance(messages[i], HumanMessage):
            last_human_message = messages[i]
            break

    summary_context_messages = []
    if last_human_message:
        summary_context_messages.append(last_human_message)
    else:
        logger.warning("No HumanMessage found prior to ToolMessage for summarization context.")

    summary_context_messages.append(last_tool_message)

    # Append the explicit summarization instruction to the LLM
    summary_context_messages.append(HumanMessage(content="""Given the user's query, and the JSON output as the basis of answer or solution:
    1. Provide a concise answer/report to the user based on the JSON result, if it contains meaningful content
    2. Otherwise, report the error or explain why the result is not informative enough
    3. Do NOT ask for more information, propose new actions, or reinterpret the user's original request. """))

    # response = model.invoke(summary_context_messages)

    # logger.info(f"LLM Response in llm_summarize_tool_output: {response}")

    # final_ai_message_content = response.content if response.content else "I have processed the request."
    final_ai_message_content = last_tool_message.content

    new_state = {**state}
    new_state["messages"] = state["messages"] + [AIMessage(content=final_ai_message_content)]
    new_state["current_tool_call"] = None

    return new_state


def extract_parameters(state: AgentState) -> AgentState:
    """
    Extracts parameters from the latest user message and updates the state.
    Also identifies the requested action. This uses an LLM call for robust parsing.
    Handles direct numerical inputs for selections/IDs when a question is pending.
    """
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        return state # Only process human messages

    user_input_content = last_message.content.strip()
    new_state = {**state} # Start with a copy

    # --- Special Handling for Direct Numerical Answers to Pending Questions ---
    if new_state.get("pending_questions"):
        if "Please select a booking by its number." in new_state["pending_questions"]:
            try:
                selection_index = int(user_input_content) - 1
                bookings = new_state.get("bookings_summary", [])
                if 0 <= selection_index < len(bookings):
                    selected_booking = bookings[selection_index]
                    new_state["selected_booking_reference"] = selected_booking["booking_reference"]
                    new_state["extracted_params"]["restaurant_name"] = selected_booking["restaurant_name"]
                    new_state["pending_questions"] = [] # Clear pending question - success!
                    logger.info(f"User selected booking: {selected_booking['booking_reference']}")
                    return new_state
                else:
                    new_state["messages"].append(AIMessage(content="That's not a valid selection. Please choose a number from the list."))
                    # Keep pending_questions set to indicate re-prompt needed
                    return new_state
            except ValueError:
                new_state["messages"].append(AIMessage(content="I didn't understand your selection. Please provide the number of the booking."))
                # Keep pending_questions set
                return new_state

        elif "Please provide the ID of the cancellation reason." in new_state["pending_questions"]:
            try:
                cancellation_id = int(user_input_content)
                reasons = new_state.get("cancellation_reasons", [])
                if any(r.get("id") == cancellation_id for r in reasons):
                    new_state["extracted_params"]["cancellation_reason_id"] = cancellation_id
                    new_state["pending_questions"] = [] # Clear pending question - success!
                    logger.info(f"User selected cancellation reason ID: {cancellation_id}")
                    return new_state
                else:
                    new_state["messages"].append(AIMessage(content="That's not a valid cancellation reason ID. Please choose an ID from the list."))
                    # Keep pending_questions set
                    return new_state
            except ValueError:
                new_state["messages"].append(AIMessage(content="I didn't understand. Please provide a valid number for the cancellation reason ID."))
                # Keep pending_questions set
                return new_state

    # --- General LLM-based Parameter Extraction (if no direct answer was handled) ---
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
    - For `customer_bookings_and_restaurants_summary` (implied by `list_customer_bookings`), extract a list of strings named `sql_conditions`, each being a valid SQL condition for the `bookings` table or the `restaurants` table. Example: `["bookings.party_size > 2", "restaurant.name = 'Italian7'"]`. If no specific conditions, set `sql_conditions` to `[]`.
    - Allowed filter variables for `restaurants` table: 'name', 'microsite_name'
    - Allowed filter variables for `bookings` table: 'visit_date', 'visit_time', 'party_size', 'channel_code', 'status', 'channel_code', 'special_requests', 'is_leave_time_confirmed', 'room_number'
    - * None of other variables are allowed to be added to the SQL statement. 
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
    extraction_prompt = extraction_system_prompt + f"User message: {user_input_content}"

    try:
        response_parsing = model.invoke([HumanMessage(content=extraction_prompt)])
        raw_content = response_parsing.content.strip()

        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
        json_string = match.group(1) if match else raw_content

        parsed_output = json.loads(json_string)
        new_extracted_params = parsed_output.get("extracted_params", {})
        requested_action_from_llm = parsed_output.get("requested_action")

        # Update state: prioritize new extracted params
        new_state['extracted_params'].update(new_extracted_params)
        if requested_action_from_llm:
            new_state['requested_action'] = requested_action_from_llm

        # Update user_email if provided in conversation
        if "customer_email" in new_extracted_params and new_extracted_params["customer_email"]:
            new_state["user_email"] = new_extracted_params["customer_email"]

        # If a restaurant name is extracted, try to get its microsite name for future use (e.g., cancellation)
        if new_state['extracted_params'].get("restaurant_name") and not new_state.get("microsite_name"):
            restaurant_name = new_state['extracted_params']["restaurant_name"]
            try:
                restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{restaurant_name}'")
                if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                    new_state["microsite_name"] = restaurant_info[0].get("microsite_name")
                    logger.info(f"Retrieved microsite_name: {new_state['microsite_name']} for {restaurant_name}")
                else:
                    logger.warning(f"No restaurant found or invalid response for name: {restaurant_name}")
            except Exception as e:
                logger.error(f"Could not retrieve microsite_name for {restaurant_name}: {e}")

    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM extraction output (not valid JSON): {raw_content}")
        new_state["messages"].append(AIMessage(content="I had trouble understanding your request. Could you please rephrase it?"))
        new_state["requested_action"] = "none" # Reset action
    except Exception as e:
        logger.error(f"Error during parameter extraction: {e}")
        new_state["messages"].append(AIMessage(content=f"An internal error occurred during processing: {e}"))
        new_state["requested_action"] = "none" # Reset action

    return new_state

def check_missing_params(state: AgentState) -> AgentState:
    """
    Checks for missing required parameters for the requested action.
    If parameters are missing, sets pending_questions.
    """
    requested_action = state.get("requested_action")
    new_state = {**state}
    
    if not requested_action or requested_action == "none":
        return new_state # No specific action identified

    required_params = get_required_params_for_action(requested_action)
    missing_params = []

    # Special handling for actions that derive required parameters from selection or existing state
    if requested_action in ["update_booking", "cancel_booking", "check_booking"]:
        if not state.get("selected_booking_reference") and not state["extracted_params"].get("booking_reference"):
            missing_params.append("booking_reference")
        if not state["extracted_params"].get("restaurant_name"):
            missing_params.append("restaurant_name")
        if requested_action == "cancel_booking":
            if not state.get("microsite_name"):
                missing_params.append("microsite_name")
            if state["extracted_params"].get("cancellation_reason_id") is None:
                missing_params.append("cancellation_reason_id")
    else: # For other actions (find_availability, make_booking, find_restaurants) check directly from extracted_params
        for param in required_params:
            if get_extracted_value(state, param) is None:
                missing_params.append(param)
    
    new_state["pending_questions"] = [] # Clear previous questions before setting new ones

    if missing_params:
        question_parts = [p.replace('_', ' ') for p in missing_params]
        if len(question_parts) == 1:
            new_state["pending_questions"].append(f"What is the {question_parts[0]}?")
        else:
            new_state["pending_questions"].append(f"I need the following information: {', '.join(question_parts)}.")
    
    return new_state


def ask_for_clarification(state: AgentState) -> AgentState:
    """Generates a message asking the user for missing information."""
    questions = state.get("pending_questions", ["What else can I help you with?"])
    response_content = " ".join(questions)
    
    new_state = {**state}
    new_state["messages"] = state["messages"] + [AIMessage(content=response_content)]
    
    return new_state

def get_customer_bookings(state: AgentState) -> AgentState:
    """Fetches customer bookings using AIToolCallingInterface."""
    user_email = state.get("user_email")
    new_state = {**state}
    
    if not user_email:
        new_state["pending_questions"] = ["I need your email address to retrieve your bookings."]
        return new_state

    sql_conditions = new_state["extracted_params"].get("sql_conditions", []) # Use new_state for current params

    try:
        bookings = AIToolCallingInterface.customer_bookings_and_restaurants_summary(user_email, sql_conditions)
        if isinstance(bookings, dict) and "error" in bookings:
            raise Exception(bookings["error"])

        new_state["bookings_summary"] = bookings
        logger.info(f"Fetched customer bookings: {bookings}")
    except Exception as e:
        logger.error(f"Error fetching customer bookings: {e}")
        new_state["messages"] = state["messages"] + [AIMessage(content=f"An error occurred while fetching your bookings: {e}")]
        new_state["requested_action"] = None # Clear action on error
        new_state["bookings_summary"] = [] # Clear any partial data

    return new_state

def handle_multiple_bookings(state: AgentState) -> AgentState:
    """Presents multiple bookings to the user and asks for selection."""
    bookings = state.get("bookings_summary", [])
    new_state = {**state}
    
    if not bookings: # This should ideally be caught before this node, but as a safeguard.
        new_state["messages"] = state["messages"] + [AIMessage(content="No bookings found to display for selection.")]
        new_state["requested_action"] = None # Clear selected action
        new_state["pending_questions"] = [] # Clear any previous questions
        return new_state

    summary_messages = ["I found multiple bookings for you:"]
    for i, booking in enumerate(bookings):
        summary_messages.append(f"{i+1}. Booking Ref: {booking.get('booking_reference', 'N/A')}, "
                                f"Restaurant: {booking.get('restaurant_name', 'N/A')}, "
                                f"Date: {booking.get('visit_date', 'N/A')}, "
                                f"Time: {booking.get('visit_time', 'N/A')}, "
                                f"Party Size: {booking.get('party_size', 'N/A')}")
    
    summary_messages.append("Please tell me the number of the booking you'd like to proceed with (e.g., '1').")
    new_state["pending_questions"] = ["Please select a booking by its number."] # Set this so extract_parameters knows what to expect
    new_state["messages"] = state["messages"] + [AIMessage(content="\n".join(summary_messages))]
    return new_state


def get_cancellation_reasons(state: AgentState) -> AgentState:
    """Fetches cancellation reasons and asks user to choose."""
    new_state = {**state}
    
    try:
        reasons = AIToolCallingInterface.list_cancellation_reasons()
        if isinstance(reasons, dict) and "error" in reasons:
            raise Exception(reasons["error"])

        new_state["cancellation_reasons"] = reasons
        if reasons:
            reason_messages = ["Please select a cancellation reason by its ID:"]
            for r in reasons:
                reason_messages.append(f"ID: {r.get('id', 'N/A')} - Reason: {r.get('reason', 'N/A')}: {r.get('description', '')}")

            new_state["pending_questions"] = ["Please provide the ID of the cancellation reason."]
            new_state["messages"] = state["messages"] + [AIMessage(content="\n".join(reason_messages))]
        else:
            new_state["messages"] = state["messages"] + [AIMessage(content="Could not retrieve cancellation reasons. Please try again.")]
            new_state["requested_action"] = None # Clear action on error
    except Exception as e:
        logger.error(f"Error fetching cancellation reasons: {e}")
        new_state["messages"] = state["messages"] + [AIMessage(content=f"An error occurred while fetching cancellation reasons: {e}")]
        new_state["requested_action"] = None # Clear action on error
    
    return new_state


def call_tool(state: AgentState) -> AgentState:
    """Executes the identified tool call stored in state."""
    tool_call_info = state.get("current_tool_call")
    if not tool_call_info:
        logger.warning("call_tool node called without a current_tool_call. This should not happen if routing is correct.")
        new_state = {**state}
        new_state["messages"] = state["messages"] + [AIMessage(content="Internal error: No tool call information found.")]
        return new_state

    tool_name = tool_call_info["name"]
    tool_args = tool_call_info["args"]
    tool_id = tool_call_info.get("id", f"auto_generated_id_{datetime.now().timestamp()}")

    selected_tool = next((t for t in tools if t.name == tool_name), None)

    new_state = {**state}

    if not selected_tool:
        logger.error(f"Attempted to call unknown tool: {tool_name}")
        new_state["messages"] = state["messages"] + [AIMessage(content=f"Error: Attempted to call an unknown function: {tool_name}.")]
        new_state["current_tool_call"] = None # Clear tool call
        return new_state

    try:
        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
        tool_output = selected_tool.invoke(tool_args)
        logger.info(f"Raw Tool output for {tool_name}: {tool_output}")

        if isinstance(tool_output, (dict, list)):
            tool_content = json.dumps(tool_output, indent=2)
        else:
            tool_content = str(tool_output)

        new_state["messages"] = state["messages"] + [ToolMessage(content=tool_content, tool_call_id=tool_id)]
        new_state["current_tool_call"] = None # Clear tool call after execution
        return new_state
    except Exception as e:
        logger.error(f"Error calling tool {tool_name} with args {tool_args}: {e}")
        error_message = f"An error occurred while executing {tool_name}: {e}"
        new_state["messages"] = state["messages"] + [ToolMessage(content=error_message, tool_call_id=tool_id)]
        new_state["current_tool_call"] = None # Clear tool call
        return new_state


def perform_action(state: AgentState) -> AgentState:
    """
    This node serves as a dispatcher to prepare arguments and set `current_tool_call`
    for the appropriate tool based on the `requested_action`.
    """
    requested_action = state.get("requested_action")
    extracted_params = state.get("extracted_params", {})
    user_email = state.get("user_email")
    selected_booking_reference = state.get("selected_booking_reference")
    microsite_name = state.get("microsite_name")

    args_for_tool = {}
    tool_to_call_name = None
    
    new_state = {**state}

    # Ensure email is present for operations that require it (e.g., list_customer_bookings, create_booking)
    if not user_email and requested_action in ["list_customer_bookings", "make_booking"]:
        new_state["messages"] = state["messages"] + [AIMessage(content="I need your email address to perform this action.")]
        new_state["requested_action"] = None
        new_state["pending_questions"] = ["What is your email address?"] # Will route to ask_for_clarification
        return new_state

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
            "customer_email": user_email,
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
        args_for_tool = {k: v for k, v in all_create_booking_args.items() if v is not None}

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
            "booking_reference": extracted_params.get("booking_reference") or selected_booking_reference
        }
        if not args_for_tool["restaurant_name"] or not args_for_tool["booking_reference"]:
            new_state["messages"] = state["messages"] + [AIMessage(content="To check a specific booking, I need both the restaurant name and the booking reference.")]
            new_state["requested_action"] = None
            return new_state

    elif requested_action == "update_booking":
        tool_to_call_name = update_booking_details_tool.name
        args_for_tool = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "booking_reference": selected_booking_reference if selected_booking_reference else extracted_params.get("booking_reference")
        }
        update_params = ["visit_date", "visit_time", "party_size", "special_requests", "is_leave_time_confirmed"]
        updates_provided = False
        for param in update_params:
            if extracted_params.get(param) is not None:
                args_for_tool[param] = extracted_params[param]
                updates_provided = True
        
        if not updates_provided:
            new_state["messages"].append(AIMessage(content="What details would you like to update (e.g., date, time, party size, special requests)?"))
            new_state["pending_questions"].append("What details would you like to update?")
            new_state["requested_action"] = "update_booking" # Keep the action to re-enter this flow
            return new_state

        if not args_for_tool["restaurant_name"] or not args_for_tool["booking_reference"]:
            new_state["messages"] = state["messages"] + [AIMessage(content="To update a booking, I need the restaurant name and the booking reference.")]
            new_state["requested_action"] = None
            return new_state

    elif requested_action == "cancel_booking":
        tool_to_call_name = cancel_booking_tool.name
        args_for_tool = {
            "restaurant_name": extracted_params.get("restaurant_name"),
            "booking_reference": selected_booking_reference if selected_booking_reference else extracted_params.get("booking_reference"),
            "microsite_name": microsite_name,
            "cancellation_reason_id": extracted_params.get("cancellation_reason_id")
        }
        if not args_for_tool["restaurant_name"] or not args_for_tool["booking_reference"] or not args_for_tool["microsite_name"] or args_for_tool["cancellation_reason_id"] is None:
            new_state["messages"] = state["messages"] + [AIMessage(content="To cancel a booking, I need the restaurant name, booking reference, microsite name, and a cancellation reason ID.")]
            new_state["requested_action"] = None
            return new_state

    elif requested_action == "find_restaurants":
        tool_to_call_name = find_restaurants_tool.name
        args_for_tool = {
            "sql_condition": extracted_params.get("sql_condition", "")
        }
        
    else:
        new_state["messages"] = state["messages"] + [AIMessage(content="I'm not sure how to perform that action based on the information provided.")]
        new_state["requested_action"] = "none" # Clear action if unable to map
        return new_state

    if tool_to_call_name:
        mock_tool_call = {
            "name": tool_to_call_name,
            "args": args_for_tool,
            "id": f"manual_call_{tool_to_call_name}_{datetime.now().timestamp()}"
        }
        new_state["current_tool_call"] = mock_tool_call
        logger.info(f"Prepared tool call: {mock_tool_call}")

    return new_state


def decide_what_to_do(state: AgentState) -> str:
    """
    Central router function for the graph, determining the next node based on the current state.
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

    logger.info(f"Router (decide_what_to_do) called. State snapshot: "
                f"action={requested_action}, pending={bool(pending_questions)}, "
                f"tool_call_prepared={bool(current_tool_call)}, "
                f"bookings_count={len(bookings_summary) if bookings_summary else 0}, "
                f"selected_ref={selected_booking_reference}, last_msg_type={type(last_message).__name__}")

    # Condition 1: Tool call was just executed, summarize its output.
    if isinstance(last_message, ToolMessage):
        logger.info("Routing to llm_summarize_tool_output (ToolMessage received).")
        return "llm_summarize_tool_output"

    # Condition 2: A tool call has been prepared and is ready for execution.
    if current_tool_call:
        logger.info("Routing to call_tool (current_tool_call exists).")
        return "call_tool"

    # Condition 3: Missing information detected or invalid input for a specific question.
    # If pending questions exist AND the last message is already an AI message giving feedback,
    # it means `extract_parameters` or a similar node already tried to handle the user's input
    # and told them it was invalid. In this case, we end the turn and wait for valid input.
    if pending_questions:
        if isinstance(last_message, AIMessage) and any(err_text in last_message.content for err_text in ["That's not a valid", "I didn't understand", "I had trouble understanding", "What details would you like to update?"]):
            logger.info("Ending turn (pending_questions present, but AI already gave feedback).")
            return END # End the turn here, wait for user's next input which restarts at extract_parameters.
        else:
            # If pending questions exist but no immediate error message, ask for clarification.
            logger.info("Routing to ask_for_clarification (pending_questions exist).")
            return "ask_for_clarification"

    # From this point onwards, `pending_questions` should be empty, meaning we have to determine a new action
    # or progress an existing action for which initial parameters might be available.

    # Condition 4: Handle multi-step actions (e.g., booking management, cancellation)

    # If an action like check/update/cancel/list bookings is requested, and we haven't selected one yet.
    if requested_action in ["list_customer_bookings", "check_booking", "update_booking", "cancel_booking"]:
        # If no specific booking identified yet (either by selection or direct input)
        if not selected_booking_reference and (not extracted_params.get("booking_reference") or not extracted_params.get("restaurant_name")):
            if not bookings_summary: # Need to fetch bookings
                logger.info("Routing to get_customer_bookings (action requires booking, none fetched).")
                return "get_customer_bookings"
            elif len(bookings_summary) > 1: # Multiple bookings, need user selection
                logger.info("Routing to handle_multiple_bookings (multiple bookings found).")
                return "handle_multiple_bookings"
            elif len(bookings_summary) == 1: # Single booking, auto-select
                selected_booking = bookings_summary[0]
                state["selected_booking_reference"] = selected_booking["booking_reference"]
                state["extracted_params"]["restaurant_name"] = selected_booking["restaurant_name"]
                # For cancellation, also get microsite name if possible
                if requested_action == "cancel_booking" and not state.get("microsite_name"):
                    try:
                        restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{selected_booking['restaurant_name']}'")
                        if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                            state["microsite_name"] = restaurant_info[0].get("microsite_name")
                            logger.info(f"Auto-selected microsite_name: {state['microsite_name']} for {selected_booking['restaurant_name']}")
                        else:
                            logger.warning(f"No restaurant found or invalid response for name: {selected_booking['restaurant_name']}")
                    except Exception as e:
                        logger.warning(f"Could not retrieve microsite_name for {selected_booking['restaurant_name']}: {e}")
                logger.info("Auto-selected single booking, routing to check_missing_params.")
                return "check_missing_params" # Now that booking is selected, check if other params are missing.
            else: # No bookings found after fetching
                state["messages"].append(AIMessage(content="I couldn't find any bookings matching that criteria for your email."))
                state["requested_action"] = None # Reset action
                state["bookings_summary"] = [] # Clear summary
                logger.info("No bookings found, ending turn.")
                return END # End the turn with this message.

        # If a booking is selected for cancellation, and cancellation reason is needed
        if requested_action == "cancel_booking" and state["extracted_params"].get("cancellation_reason_id") is None:
            if not cancellation_reasons: # Need to fetch reasons
                logger.info("Routing to get_cancellation_reasons (cancellation reason needed).")
                return "get_cancellation_reasons"
            logger.info("Cancellation reasons already fetched or user needs to provide ID; routing to check_missing_params.")
            return "check_missing_params"

    # Condition 5: A specific action has been requested (and no multi-step issues remain).
    # Check if all parameters for this action are now available.
    if requested_action and requested_action != "none":
        logger.info(f"Routing to check_missing_params (action '{requested_action}' identified).")
        return "check_missing_params"

    # Condition 6: No clear action identified by `extract_parameters`, or initial state.
    # Let the LLM decide what to do or respond generally.
    logger.info("Routing to call_llm (no specific action or pending questions).")
    return "call_llm"


# --- LangGraph Setup ---
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("call_llm", call_llm)
workflow.add_node("llm_summarize_tool_output", llm_summarize_tool_output)
workflow.add_node("extract_parameters", extract_parameters)
workflow.add_node("check_missing_params", check_missing_params)
workflow.add_node("ask_for_clarification", ask_for_clarification)
workflow.add_node("get_customer_bookings", get_customer_bookings)
workflow.add_node("handle_multiple_bookings", handle_multiple_bookings)
workflow.add_node("get_cancellation_reasons", get_cancellation_reasons)
workflow.add_node("call_tool", call_tool)
workflow.add_node("perform_action", perform_action)
# 'decide_what_to_do' is purely a conditional router, NOT a node itself.

# Define edges
# Entry point: Always start by extracting parameters from the user's message
workflow.set_entry_point("extract_parameters")

# After extract_parameters, decide what to do next based on the state.
workflow.add_conditional_edges(
    "extract_parameters",
    decide_what_to_do, # This is the routing function
    {
        "llm_summarize_tool_output": "llm_summarize_tool_output",
        "call_tool": "call_tool",
        "ask_for_clarification": "ask_for_clarification",
        END: END,
        "get_customer_bookings": "get_customer_bookings",
        "handle_multiple_bookings": "handle_multiple_bookings",
        "get_cancellation_reasons": "get_cancellation_reasons",
        "check_missing_params": "check_missing_params",
        "perform_action": "perform_action",
        "call_llm": "call_llm",
    }
)

# If more clarification is needed, it will loop back to extract_parameters after the user's next input.
workflow.add_edge("ask_for_clarification", "extract_parameters")

# After fetching customer bookings, return to the central router to decide what to do next.
workflow.add_conditional_edges(
    "get_customer_bookings",
    decide_what_to_do, # Routing function
    {
        "llm_summarize_tool_output": "llm_summarize_tool_output", # Should not happen from here, but for completeness
        "call_tool": "call_tool",
        "ask_for_clarification": "ask_for_clarification",
        END: END,
        "handle_multiple_bookings": "handle_multiple_bookings", # If multiple found
        "check_missing_params": "check_missing_params", # If single auto-selected
        "call_llm": "call_llm", # If no bookings found after fetch and no clear error message
        "get_customer_bookings": "get_customer_bookings", # Fallback, unlikely but possible for state re-evaluation
        "get_cancellation_reasons": "get_cancellation_reasons",
        "perform_action": "perform_action",
    }
)


# After presenting multiple bookings, the user's next input should be a selection, so back to extract_parameters.
workflow.add_edge("handle_multiple_bookings", "extract_parameters")

# After getting cancellation reasons, the user's next input should be an ID, so back to extract_parameters.
workflow.add_edge("get_cancellation_reasons", "extract_parameters")

# Once `perform_action` prepares a tool call, it transitions to `call_tool`.
workflow.add_edge("perform_action", "call_tool")

# After `call_tool` executes, the result needs to be summarized.
workflow.add_edge("call_tool", "llm_summarize_tool_output")

# After summarization, the turn is complete.
workflow.add_edge("llm_summarize_tool_output", END)

# After `check_missing_params`, the next step depends on whether questions are pending or not.
workflow.add_conditional_edges(
    "check_missing_params",
    lambda state: "ask_for_clarification" if state["pending_questions"] else "perform_action",
    {
        "ask_for_clarification": "ask_for_clarification",
        "perform_action": "perform_action"
    }
)

# After `call_llm`, use decide_what_to_do as the router
workflow.add_conditional_edges(
    "call_llm",
    decide_what_to_do, # Routing function
    {
        "llm_summarize_tool_output": "llm_summarize_tool_output",
        "call_tool": "call_tool",
        "ask_for_clarification": "ask_for_clarification",
        END: END,
        "get_customer_bookings": "get_customer_bookings",
        "handle_multiple_bookings": "handle_multiple_bookings",
        "get_cancellation_reasons": "get_cancellation_reasons",
        "check_missing_params": "check_missing_params",
        "perform_action": "perform_action",
        "call_llm": "call_llm", # If LLM decides to just chat further or needs more info from itself
    }
)

# Compile the graph
graph = workflow.compile()

# Initial state for the agent
initial_agent_state = AgentState(
    messages=[
        HumanMessage(content="""You are an AI assistant for restaurant bookings.
        When you receive tool output, summarize the results in a user-friendly way. Do not output raw JSON or formatted HTML directly.
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
    print("Note: Structured data like booking lists and availability will now be summarized in plain text.")
    print("\nHow can I help you today?")

    current_state = copy.deepcopy(initial_state)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        current_state['messages'].append(HumanMessage(content=user_input))

        current_state["current_tool_call"] = None
        
        # Reset specific action-related data only if the user is initiating a *new* primary action
        if current_state.get("requested_action") is None or user_input.strip().lower() in ["hi", "hello", "new query", "start over"]:
            current_state["bookings_summary"] = []
            current_state["selected_booking_reference"] = None
            current_state["cancellation_reasons"] = []
            current_state["microsite_name"] = None
            current_state["extracted_params"] = {}


        try:
            final_step_output = None
            for step_output in graph.stream(current_state):
                final_step_output = step_output
                logger.debug(f"Graph step output: {step_output}")
            
            if final_step_output:
                current_state = list(final_step_output.values())[0]
            else:
                logger.error("Graph stream returned no states")
                print("AI: I encountered an issue processing your request. Let me try again.")
                current_state = copy.deepcopy(initial_agent_state)
                current_state['user_email'] = initial_state['user_email']
                continue

            if 'messages' not in current_state or not current_state['messages']:
                logger.error("Final state missing messages: " + str(current_state))
                print("AI: I'm sorry, I encountered an internal issue. Please try again.")
                current_state = copy.deepcopy(initial_agent_state)
                current_state['user_email'] = initial_state['user_email']
                continue

            final_message = current_state['messages'][-1]
            if isinstance(final_message, AIMessage):
                print(f"AI: {final_message.content}")
            elif isinstance(final_message, ToolMessage):
                print(f"AI: I've processed your request. The raw result: {final_message.content}")
            else:
                print(f"AI: {getattr(final_message, 'content', 'I am processing your request.')}")

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print(f"AI: An unexpected error occurred: {e}. Please try again or rephrase your request.")
            current_state = copy.deepcopy(initial_agent_state)
            current_state['user_email'] = initial_state['user_email']
            current_state['messages'].append(AIMessage(content="I'm sorry, I encountered an error. Let's start fresh. How can I help you?"))

if __name__ == "__main__":
    run_chat_loop(initial_agent_state)
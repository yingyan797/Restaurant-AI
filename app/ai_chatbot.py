import os, sys, json
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from datetime import datetime, date, time
import re # Import the regex module for stripping markdown
import copy

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = ""

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


# In your ai_chatbot.py file, locate and update the llm_summarize_tool_output function:

def llm_summarize_tool_output(state: AgentState) -> AgentState:
    messages = state['messages'] # Get the full message history

    # The last message is guaranteed to be the ToolMessage
    last_tool_message = messages[-1]

    # Find the most recent HumanMessage in the history.
    # This is the message that initiated the sequence leading to the tool call.
    last_human_message = None
    for i in range(len(messages) - 2, -1, -1): # Iterate backwards from before the ToolMessage
        if isinstance(messages[i], HumanMessage):
            last_human_message = messages[i]
            break

    # Prepare the specific messages to send to the LLM for summarization.
    # We want: [triggering_human_message, tool_output_message, summarization_instruction]
    summary_context_messages = []
    if last_human_message:
        summary_context_messages.append(last_human_message)
    else:
        # Fallback: If for some reason there's no preceding HumanMessage (unlikely in a normal flow),
        # start with a generic context or just the tool message.
        # For a chatbot, a tool call usually implies a user interaction.
        logger.warning("No HumanMessage found prior to ToolMessage for summarization context.")

    summary_context_messages.append(last_tool_message)

    # Append the explicit summarization instruction to the LLM
    summary_context_messages.append(HumanMessage(content="""Given the user's query, and the JSON output as the basis of answer or solution:
    1. Provide a concise answer/report to the user based on the JSON result, if it contains meaningful content
    2. Otherwise, report the error or explain why the result is not informative enough
    3. Do NOT ask for more information, propose new actions, or reinterpret the user's original request. """))

    # print("Human:", last_human_message, "Tool:", last_tool_message)
    # response = model.invoke(summary_context_messages) # Use the limited context here

    # logger.info(f"LLM Response in llm_summarize_tool_output: {response}")

    # Extract content. If the LLM still tries to output a tool_call (undesirable, but guard against it),
    # we take its content and ensure current_tool_call is None.
    # final_ai_message_content = response.content if response.content else "I have processed the request."
    final_ai_message_content = last_tool_message.content

    # Create new state: IMPORTANT - add the AI's summary message to the FULL conversation history.
    # The 'messages' in the state should always represent the complete conversation for proper
    # context in subsequent turns (e.g., for 'extract_parameters' to understand the latest user input).
    new_state = {**state}
    new_state["messages"] = state["messages"] + [AIMessage(content=final_ai_message_content)]
    new_state["current_tool_call"] = None

    return new_state

def extract_parameters(state: AgentState) -> AgentState:
    """
    Extracts parameters from the latest user message and updates the state.
    Also identifies the requested action. This uses an LLM call for robust parsing.
    """
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        return state # Only process human messages

    # System prompt for LLM to extract intent and parameters
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

    # Create new state by copying existing state
    new_state = {**state}

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
        new_state['extracted_params'].update(new_extracted_params)
        if requested_action_from_llm:
            new_state['requested_action'] = requested_action_from_llm

        # Update user_email if provided in conversation
        if "customer_email" in new_extracted_params and new_extracted_params["customer_email"]:
            new_state["user_email"] = new_extracted_params["customer_email"]

        # If a restaurant name is extracted, try to get its microsite name for future use (e.g., cancellation)
        # This is a pre-emptive fetch, it should ideally happen only once per restaurant or when needed.
        if new_state['extracted_params'].get("restaurant_name") and not new_state.get("microsite_name"):
            restaurant_name = new_state['extracted_params']["restaurant_name"]
            try:
                # Using a direct SQL condition for exact match
                restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{restaurant_name}'")
                if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                    new_state["microsite_name"] = restaurant_info[0].get("microsite_name")
                    logger.info(f"Retrieved microsite_name: {new_state['microsite_name']} for {restaurant_name}")
                else:
                    logger.warning(f"No restaurant found or invalid response for name: {restaurant_name}")
            except Exception as e:
                logger.error(f"Could not retrieve microsite_name for {restaurant_name}: {e}")

    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM extraction output (not valid JSON): {raw_content}") # Log raw content for debugging
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
    
    new_state["pending_questions"] = [] # Clear previous questions

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
        return new_state # Routing will send to ask_for_clarification

    sql_conditions = state["extracted_params"].get("sql_conditions", [])

    try:
        bookings = AIToolCallingInterface.customer_bookings_and_restaurants_summary(user_email, sql_conditions)
        if isinstance(bookings, dict) and "error" in bookings:
            raise Exception(bookings["error"])

        new_state["bookings_summary"] = bookings
        logger.info(f"Fetched customer bookings: {bookings}")
    except Exception as e:
        logger.error(f"Error fetching customer bookings: {e}")
        new_state["messages"] = state["messages"] + [AIMessage(content=f"An error occurred while fetching your bookings: {e}")]

    return new_state

def handle_multiple_bookings(state: AgentState) -> AgentState:
    """Presents multiple bookings to the user and asks for selection."""
    bookings = state.get("bookings_summary", [])
    new_state = {**state}
    
    if not bookings:
        new_state["messages"] = state["messages"] + [AIMessage(content="No bookings found to display for selection.")]
        # Clear selected action as we can't proceed without a booking.
        new_state["requested_action"] = None
        new_state["pending_questions"] = []
        return new_state

    summary_messages = ["I found multiple bookings for you:"]
    # Instead of HTML table, list them as plain text
    for i, booking in enumerate(bookings):
        summary_messages.append(f"{i+1}. Booking Ref: {booking.get('booking_reference', 'N/A')}, "
                                f"Restaurant: {booking.get('restaurant_name', 'N/A')}, "
                                f"Date: {booking.get('visit_date', 'N/A')}, "
                                f"Time: {booking.get('visit_time', 'N/A')}, "
                                f"Party Size: {booking.get('party_size', 'N/A')}")
    
    summary_messages.append("Please tell me the number of the booking you'd like to proceed with (e.g., '1').")
    new_state["pending_questions"] = ["Please select a booking by its number."]
    new_state["messages"] = state["messages"] + [AIMessage(content="\n".join(summary_messages))]
    return new_state

def process_selected_booking(state: AgentState) -> AgentState:
    """
    Processes the user's selection for a booking from a list.
    Assumes the last message contains a number.
    """
    last_message_content = state['messages'][-1].content
    bookings = state.get("bookings_summary", [])
    
    new_state = {**state}

    try:
        selection_index = int(last_message_content) - 1
        if 0 <= selection_index < len(bookings):
            selected_booking = bookings[selection_index]
            new_state["selected_booking_reference"] = selected_booking["booking_reference"]
            new_state["extracted_params"]["restaurant_name"] = selected_booking["restaurant_name"]
            # Clear pending questions as selection is made
            new_state["pending_questions"] = []
            logger.info(f"User selected booking: {selected_booking['booking_reference']}")

            # Try to get microsite name for the selected restaurant if not already present
            if not state.get("microsite_name") and new_state["extracted_params"].get("restaurant_name"):
                restaurant_name = new_state["extracted_params"]["restaurant_name"]
                try:
                    db_session = AIToolCallingInterface.get_db_session()
                    if db_session:
                        restaurant_info = AIToolCallingInterface.find_restaurants(f"WHERE name='{restaurant_name}'")
                        if restaurant_info and not isinstance(restaurant_info, dict) and len(restaurant_info) > 0:
                            new_state["microsite_name"] = restaurant_info[0].get("microsite_name")
                            logger.info(f"Retrieved microsite_name: {new_state['microsite_name']} for {restaurant_name}")
                        db_session.close()
                except Exception as e:
                    logger.warning(f"Could not retrieve microsite_name for {restaurant_name}: {e}")

        else:
            response_content = "That's not a valid selection. Please choose a number from the list."
            new_state["messages"] = state["messages"] + [AIMessage(content=response_content)]
            new_state["pending_questions"] = ["Please select a valid booking number."] # Re-ask
    except ValueError:
        response_content = "I didn't understand your selection. Please provide the number of the booking."
        new_state["messages"] = state["messages"] + [AIMessage(content=response_content)]
        new_state["pending_questions"] = ["Please provide a number."] # Re-ask

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
            # Instead of HTML table, list them as plain text
            for r in reasons:
                reason_messages.append(f"ID: {r.get('id', 'N/A')} - Reason: {r.get('reason', 'N/A')}: {r.get('description', '')}")

            new_state["pending_questions"] = ["Please provide the ID of the cancellation reason."]
            new_state["messages"] = state["messages"] + [AIMessage(content="\n".join(reason_messages))]
        else:
            new_state["messages"] = state["messages"] + [AIMessage(content="Could not retrieve cancellation reasons. Please try again.")]
    except Exception as e:
        logger.error(f"Error fetching cancellation reasons: {e}")
        new_state["messages"] = state["messages"] + [AIMessage(content=f"An error occurred while fetching cancellation reasons: {e}")]
    
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
    tool_id = tool_call_info.get("id", f"auto_generated_id_{datetime.now().timestamp()}") # Get ID or use a dummy

    # Find the LangChain tool object by name
    selected_tool = next((t for t in tools if t.name == tool_name), None)

    new_state = {**state}

    if not selected_tool:
        logger.error(f"Attempted to call unknown tool: {tool_name}")
        new_state["messages"] = state["messages"] + [AIMessage(content=f"Error: Attempted to call an unknown function: {tool_name}.")]
        return new_state

    try:
        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
        tool_output = selected_tool.invoke(tool_args)
        logger.info(f"Raw Tool output for {tool_name}: {tool_output}")

        # No special handling for HTML rendering; output is always JSON or string
        if isinstance(tool_output, (dict, list)):
            tool_content = json.dumps(tool_output, indent=2) # Pretty print JSON for readability in logs/debug
        else:
            tool_content = str(tool_output)

        new_state["messages"] = state["messages"] + [ToolMessage(content=tool_content, tool_call_id=tool_id)]
        return new_state
    except Exception as e:
        logger.error(f"Error calling tool {tool_name} with args {tool_args}: {e}")
        error_message = f"An error occurred while executing {tool_name}: {e}"
        # Return a ToolMessage with error content so LLM can be aware
        new_state["messages"] = state["messages"] + [ToolMessage(content=error_message, tool_call_id=tool_id)]
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
    microsite_name = state.get("microsite_name") # Crucial for cancel

    args_for_tool = {}
    tool_to_call_name = None
    
    new_state = {**state}

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
        for param in update_params:
            if extracted_params.get(param) is not None:
                args_for_tool[param] = extracted_params[param]
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
        # Changed default sql_condition to an empty string as requested
        args_for_tool = {
            "sql_condition": extracted_params.get("sql_condition", "") # Default to empty string
        }
        
    else:
        new_state["messages"] = state["messages"] + [AIMessage(content="I'm not sure how to perform that action based on the information provided.")]
        new_state["requested_action"] = None # Clear action if unable to map
        return new_state

    if tool_to_call_name:
        # Manually create a tool call object to pass to the call_tool node
        mock_tool_call = {
            "name": tool_to_call_name,
            "args": args_for_tool,
            "id": f"manual_call_{tool_to_call_name}_{datetime.now().timestamp()}" # Unique ID
        }
        new_state["current_tool_call"] = mock_tool_call
        logger.info(f"Prepared tool call: {mock_tool_call}")

    return new_state # Router will pick up current_tool_call and route to call_tool


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
    #    Now, route to the summarization node.
    if isinstance(last_message, ToolMessage):
        return "llm_summarize_tool_output"

    # 2. If there are pending questions, it means we just asked for info.
    #    Now we expect the user to have provided it, so route to process that input.
    if pending_questions:
        if "Please select a booking by its number." in pending_questions:
            return "process_selected_booking"
        elif "Please provide the ID of the cancellation reason." in pending_questions:
            # User provided a number for cancellation ID. Try to parse it directly.
            # This logic should be handled by `extract_parameters` upon receiving the numerical input.
            return "extract_parameters" # Re-process user input to get the number/ID
        else:
            # General missing params, go back to extraction to capture user's answer
            return "extract_parameters"

    # 3. If a tool call has been prepared (either by `call_llm` or `perform_action`)
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
workflow.add_node("llm_summarize_tool_output", llm_summarize_tool_output) # New node
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

# After asking for clarification, the next user message will be processed by extract_parameters
workflow.add_edge("ask_for_clarification", "extract_parameters")

# After getting customer bookings, route based on the number of bookings found
workflow.add_conditional_edges(
    "get_customer_bookings",
    decide_next_step,
    {
        "call_llm": "call_llm", # If no bookings found, LLM generates a response
        "handle_multiple_bookings": "handle_multiple_bookings",
        "check_missing_params": "check_missing_params", # If single booking auto-selected
        END: END # Should not typically end here without LLM response
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

# After `call_tool` executes, feed the tool output to the new summarization node
workflow.add_edge("call_tool", "llm_summarize_tool_output")

# After `llm_summarize_tool_output`, the turn is considered complete.
workflow.add_edge("llm_summarize_tool_output", END)

# Main conditional routing logic
# This edge determines the flow after `extract_parameters`, and other nodes that need to re-evaluate state.
workflow.add_conditional_edges(
    "extract_parameters",
    decide_next_step, # The router function
    {
        "call_llm": "call_llm", # LLM to decide
        "extract_parameters": "extract_parameters", # For recursive extraction or re-evaluation after user input
        "check_missing_params": "check_missing_params",
        "ask_for_clarification": "ask_for_clarification",
        "get_customer_bookings": "get_customer_bookings",
        "handle_multiple_bookings": "handle_multiple_bookings",
        "process_selected_booking": "process_selected_booking",
        "get_cancellation_reasons": "get_cancellation_reasons",
        "perform_action": "perform_action",
        "call_tool": "call_tool", # If current_tool_call is already set by LLM (from call_llm node) or manual routing
        "llm_summarize_tool_output": "llm_summarize_tool_output", # This case should not happen from here, but for completeness
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

# After `call_llm` (which is for initial LLM decision),
# route based on whether it decided to call a tool or had a direct response.
workflow.add_conditional_edges(
    "call_llm",
    decide_next_step, # Re-use decide_next_step to determine if current_tool_call exists or END
    {
        "call_tool": "call_tool", # If LLM generated a tool call
        END: END # If LLM generated a direct text response
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

    # Create a deep copy of the initial state for each session to avoid state leakage
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
        # Only reset if no action is currently pending
        if current_state.get("requested_action") is None: # This logic might need refinement based on desired reset behavior
            current_state["bookings_summary"] = []
            current_state["selected_booking_reference"] = None
            current_state["cancellation_reasons"] = []
            current_state["microsite_name"] = None
            current_state["extracted_params"] = {} # Also clear extracted params for a fresh start


        try:
            # Process the conversation through the graph
            final_step_output = None
            for step_output in graph.stream(current_state):
                # Each step_output is a dictionary like {'node_name': {updated_AgentState_dict}}
                # We want the actual AgentState dictionary, which is the value of this dict.
                final_step_output = step_output
                logger.debug(f"Graph step output: {step_output}")
            
            # Extract the actual AgentState from the last yielded item
            if final_step_output:
                # Get the value of the single key-value pair, which is the full AgentState
                current_state = list(final_step_output.values())[0]
            else:
                logger.error("Graph stream returned no states")
                print("AI: I encountered an issue processing your request. Let me try again.")
                # Reset current_state if something went wrong
                current_state = copy.deepcopy(initial_agent_state)
                current_state['user_email'] = initial_state['user_email']
                continue

            # Ensure we have messages in the state
            if 'messages' not in current_state or not current_state['messages']:
                logger.error("Final state missing messages: " + str(current_state))
                print("AI: I'm sorry, I encountered an internal issue. Please try again.")
                current_state = copy.deepcopy(initial_agent_state)
                current_state['user_email'] = initial_state['user_email']
                continue

            # Get the final AI response
            final_message = current_state['messages'][-1]
            if isinstance(final_message, AIMessage):
                print(f"AI: {final_message.content}")
            elif isinstance(final_message, ToolMessage):
                # This case should ideally be handled by llm_summarize_tool_output,
                # but as a fallback, print the tool's content.
                print(f"AI: I've processed your request. The raw result: {final_message.content}")
            else:
                print(f"AI: {getattr(final_message, 'content', 'I am processing your request.')}")

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print(f"AI: An unexpected error occurred: {e}. Please try again or rephrase your request.")
            # Reset to initial state on error
            current_state = copy.deepcopy(initial_agent_state)
            current_state['user_email'] = initial_state['user_email']
            current_state['messages'].append(AIMessage(content="I'm sorry, I encountered an error. Let's start fresh. How can I help you?"))

if __name__ == "__main__":
    run_chat_loop(initial_agent_state)
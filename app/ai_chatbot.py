from datetime import date, time, datetime
from typing import Dict, Any, List, Optional, Callable, Type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError # Import ValidationError
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from app.database import get_db, Base, engine
from app.models import Restaurant, Customer, Booking, AvailabilitySlot, CancellationReason
from ai_tools import AIToolCallingInterface, logger, QueryCondition, FilterCondition # Import QueryCondition and FilterCondition dataclasses

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

    # New fields for complex queries
    query_condition: Optional[Any] # Will store QueryConditionModel, then converted to QueryCondition dataclass
    booking_conditions: Optional[Any] # Will store QueryConditionModel, then converted to QueryCondition dataclass
    restaurant_conditions: Optional[Any] # Will store QueryConditionModel, then converted to QueryCondition dataclass


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

# New Pydantic Models for Query Conditions
class FilterConditionModel(BaseModel):
    """Represents a single filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    operator: str = Field(description="The comparison operator (e.g., 'eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'in', 'not_in', 'like', 'ilike', 'is_null', 'is_not_null', 'between').")
    value: Optional[Any] = Field(None, description="The value(s) for the filter. For 'in'/'not_in' this should be a list. For 'between' this should be a list/tuple of two values. For 'is_null'/'is_not_null' this should be None.")

class QueryConditionModel(BaseModel):
    """Constraints and displayed information for database queries."""
    filters: Optional[List[FilterConditionModel]] = Field(None, description="A list of filter conditions to apply.")
    limit: Optional[int] = Field(None, description="Maximum number of records to return.")
    offset: Optional[int] = Field(None, description="Number of records to skip.")
    order_by: Optional[List[str]] = Field(None, description="List of column names to order the results by.")

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
    VisitTime: TimeStr = Field(description="The desired visit time in HH:MM:SS format.")
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
    """Get a summary of a customer's bookings and the associated restaurant information, with optional filtering."""
    email: str = Field(description="The customer's email address.")
    booking_conditions: Optional[QueryConditionModel] = Field(None, description="Optional: Conditions to filter the customer's bookings.")
    restaurant_conditions: Optional[QueryConditionModel] = Field(None, description="Optional: Conditions to filter the booked restaurants.")

class ListCancellationReasonsTool(BaseModel):
    """Get all possible cancellation reasons with their IDs, titles, and descriptions."""

class GetRestaurantsTool(BaseModel):
    """Get information of all restaurants or specific restaurants satisfying certain conditions."""
    query_condition: Optional[QueryConditionModel] = Field(None, description="Optional: Conditions to filter the restaurants.")

# Filter creation tools for advanced customer booking searches
class CreateEqualsFilterTool(BaseModel):
    """Create an equals filter condition for database queries."""
    column: str = Field(description="The column name to filter on (e.g., 'party_size', 'status', 'visit_date').")
    value: Any = Field(description="The value to match exactly.") # Changed to Any to allow diverse types

class CreateInFilterTool(BaseModel):
    """Create an IN filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    values: List[Any] = Field(description="List of values to match against.") # Changed to List[Any]

class CreateRangeFilterTool(BaseModel):
    """Create a range filter condition for database queries."""
    column: str = Field(description="The column name to filter on (e.g., 'party_size', 'visit_date').")
    lower: Any = Field(description="The lower bound of the range.") # Changed to Any
    upper: Any = Field(description="The upper bound of the range.") # Changed to Any

class CreateLikeFilterTool(BaseModel):
    """Create a LIKE filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    pattern: str = Field(description="The pattern to match (use % for wildcards).")
    case_sensitive: bool = Field(default=True, description="Whether the search should be case sensitive.")

class CreateNotEqualsFilterTool(BaseModel):
    """Create a not equals filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    value: Any = Field(description="The value to exclude.") # Changed to Any

class CreateLessThanFilterTool(BaseModel):
    """Create a less than filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    value: Any = Field(description="The value to compare against.") # Changed to Any

class CreateLessThanOrEqualFilterTool(BaseModel):
    """Create a less than or equal filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    value: Any = Field(description="The value to compare against.") # Changed to Any

class CreateGreaterThanFilterTool(BaseModel):
    """Create a greater than filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    value: Any = Field(description="The value to compare against.") # Changed to Any

class CreateGreaterThanOrEqualFilterTool(BaseModel):
    """Create a greater than or equal filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    value: Any = Field(description="The value to compare against.") # Changed to Any

class CreateNotInFilterTool(BaseModel):
    """Create a NOT IN filter condition for database queries."""
    column: str = Field(description="The column name to filter on.")
    values: List[Any] = Field(description="List of values to exclude.") # Changed to List[Any]

class CreateIsNullFilterTool(BaseModel):
    """Create an IS NULL filter condition for database queries."""
    column: str = Field(description="The column name to check for null values.")

class CreateIsNotNullFilterTool(BaseModel):
    """Create an IS NOT NULL filter condition for database queries."""
    column: str = Field(description="The column name to check for non-null values.")

TOOL_NODE_MAP = {
    "SearchAvailabilityTool": SearchAvailabilityTool, "CreateBookingTool": CreateBookingTool,
    "CancelBookingTool": CancelBookingTool, "GetBookingDetailsTool": GetBookingDetailsTool,
    "UpdateBookingDetailsTool": UpdateBookingDetailsTool,
    "GetCustomerBookingsAndRestaurantsSummaryTool": GetCustomerBookingsAndRestaurantsSummaryTool,
    "ListCancellationReasonsTool": ListCancellationReasonsTool, "GetRestaurantsTool": GetRestaurantsTool,
    "CreateEqualsFilterTool": CreateEqualsFilterTool, "CreateInFilterTool": CreateInFilterTool,
    "CreateRangeFilterTool": CreateRangeFilterTool, "CreateLikeFilterTool": CreateLikeFilterTool,
    "CreateNotEqualsFilterTool": CreateNotEqualsFilterTool, "CreateLessThanFilterTool": CreateLessThanFilterTool,
    "CreateLessThanOrEqualFilterTool": CreateLessThanOrEqualFilterTool, "CreateGreaterThanFilterTool": CreateGreaterThanFilterTool,
    "CreateGreaterThanOrEqualFilterTool": CreateGreaterThanOrEqualFilterTool, "CreateNotInFilterTool": CreateNotInFilterTool,
    "CreateIsNullFilterTool": CreateIsNullFilterTool, "CreateIsNotNullFilterTool": CreateIsNotNullFilterTool,
    # Add the new condition models to the map if they were top-level tools, but they are nested here
    "QueryConditionModel": QueryConditionModel, # Adding for potential direct parsing if needed
    "FilterConditionModel": FilterConditionModel, # Adding for potential direct parsing if needed
}

# --- LangGraph Nodes ---

# Initialize a Gemini model with tool calling capabilities
# Set your GOOGLE_API_KEY environment variable or uncomment the line below to set it directly
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

# Bind the defined tools to the Gemini model
model_with_tools = model.bind_tools([
    SearchAvailabilityTool, CreateBookingTool, CancelBookingTool, GetBookingDetailsTool,
    UpdateBookingDetailsTool, GetCustomerBookingsAndRestaurantsSummaryTool,
    ListCancellationReasonsTool, GetRestaurantsTool,
    CreateEqualsFilterTool, CreateInFilterTool, CreateRangeFilterTool, CreateLikeFilterTool,
    CreateNotEqualsFilterTool, CreateLessThanFilterTool, CreateLessThanOrEqualFilterTool,
    CreateGreaterThanFilterTool, CreateGreaterThanOrEqualFilterTool, CreateNotInFilterTool,
    CreateIsNullFilterTool, CreateIsNotNullFilterTool
])

def _convert_pydantic_to_dataclass(pydantic_model_instance: BaseModel, target_dataclass_type: Type):
    """
    Recursively converts a Pydantic model instance to its corresponding dataclass instance.
    Handles nested Pydantic models (e.g., FilterConditionModel within QueryConditionModel)
    by converting them to their respective dataclasses.
    """
    if pydantic_model_instance is None:
        return None
    
    if not isinstance(pydantic_model_instance, BaseModel):
        # If it's not a Pydantic model (e.g., already a basic type like str, int, list of str), return as is
        return pydantic_model_instance

    # Get the dictionary representation of the Pydantic model
    model_dict = pydantic_model_instance.model_dump()

    # Special handling for QueryConditionModel which has a 'filters' list
    if target_dataclass_type == QueryCondition and 'filters' in model_dict and model_dict['filters'] is not None:
        converted_filters = []
        for filter_model_data in model_dict['filters']:
            # filter_model_data might be a dict (if not already parsed to Pydantic obj) or FilterConditionModel
            if isinstance(filter_model_data, dict):
                # If it's a dict, parse it into FilterConditionModel first
                filter_model_instance = FilterConditionModel(**filter_model_data)
            else:
                filter_model_instance = filter_model_data # Assume it's already a FilterConditionModel
            converted_filters.append(_convert_pydantic_to_dataclass(filter_model_instance, FilterCondition))
        model_dict['filters'] = converted_filters
    
    # Instantiate the target dataclass
    try:
        return target_dataclass_type(**model_dict)
    except Exception as e:
        logger.error(f"Error converting Pydantic model {pydantic_model_instance.__class__.__name__} to dataclass {target_dataclass_type.__name__}: {e}")
        logger.error(f"Model dict: {model_dict}")
        return None


def _map_state_to_tool_params(state: AgentState, tool_model: BaseModel) -> Dict[str, Any]:
    """
    Maps relevant fields from the AgentState to the parameters expected by a given tool's Pydantic model.
    Handles type conversions (e.g., date/time objects to strings) where necessary.
    Also handles conversion of Pydantic QueryCondition/FilterCondition to ai_tools dataclasses.
    """
    params = {}
    for field_name, field_info in tool_model.model_fields.items():
        state_value = getattr(state, field_name, None)
        
        if state_value is None:
            continue

        # Convert date/time objects back to string format if the tool expects strings
        if field_name == "VisitDate" and isinstance(state_value, date):
            params[field_name] = state_value.isoformat()
        elif field_name == "VisitTime" and isinstance(state_value, time):
            params[field_name] = state_value.isoformat()
        elif isinstance(field_info.annotation, type) and issubclass(field_info.annotation, QueryConditionModel):
            # Convert QueryConditionModel (Pydantic) to QueryCondition (dataclass)
            params[field_name] = _convert_pydantic_to_dataclass(state_value, QueryCondition)
        elif isinstance(field_info.annotation, type) and issubclass(field_info.annotation, FilterConditionModel):
            # This case should ideally not be hit directly for top-level params, but for robustness
            params[field_name] = _convert_pydantic_to_dataclass(state_value, FilterCondition)
        else:
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
                              "For advanced searches, you can create filters to find specific customer bookings by criteria like party size, date ranges, status, etc. "
                              "Use filter tools when customers ask to find bookings/restaurants with specific conditions (e.g., 'bookings for more than 4 people', 'bookings this month', 'restaurant names starts with T'). "
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
            elif key in ["query_condition", "booking_conditions", "restaurant_conditions"] and isinstance(value, dict):
                try:
                    # Attempt to parse the dictionary into a QueryConditionModel
                    new_state[key] = QueryConditionModel(**value)
                except ValidationError as e:
                    logger.warning(f"Could not parse {key} into QueryConditionModel due to validation error: {e}"); new_state[key] = None
                except Exception as e:
                    logger.warning(f"Could not parse {key} into QueryConditionModel: {e}"); new_state[key] = None
            else:
                new_state[key] = value

        # --- NEW LOGIC: Attempt to auto-resolve booking ref/restaurant from email (or any other data) ---
        if (new_state['current_intent'] in ["UpdateBookingDetailsTool", "CancelBookingTool", "GetBookingDetailsTool"]) and \
           not (new_state.get('booking_reference') and new_state.get('restaurant_name')) and \
           new_state.get('customer_email'):
            logger.info("Missing booking ref and restaurant, using GetCustomerBookingsAndRestaurantsSummaryTool...")
            new_state['current_intent'] = "GetCustomerBookingsAndRestaurantsSummaryTool"
            new_state['missing_info'] = []  # We're calling a tool to *get* the info.
            # We don't need to clear the email, since it will be used to get the summary.

        # --- NEW LOGIC: Handle potential multiple bookings  ---
        if new_state.get('current_intent') == "GetCustomerBookingsAndRestaurantsSummaryTool" and \
           new_state.get('tool_output') and new_state.get('tool_output').get('bookings_summary'):

            bookings = new_state['tool_output']['bookings_summary']
            logger.info(f"Entering confirm_booking state.")
            booking_options_str = "\n".join([
                f"- {i+1}: {b['restaurant_name']} on {b['visit_date']} at {b['visit_time']} (Ref: {b['booking_reference']})"
                for i, b in enumerate(bookings)
            ])
            new_state['response'] = (
                "I found multiple matching bookings. Please select the booking you want to modify:\n"
                f"{booking_options_str}\n"
                "Or, type 'cancel' to cancel."
            )
            new_state['current_intent'] = "confirm_booking" # Transition to confirmation state
            new_state['missing_info'] = ['booking_reference', 'restaurant_name'] # Prompt for booking selection.
        # --- Check if we have the values after tool call, or if the user confirmed, or if the original tool name is create booking--
        elif (new_state['current_intent'] == "UpdateBookingDetailsTool" or new_state['current_intent'] == "CancelBookingTool") and \
                new_state.get('tool_output') and new_state.get('tool_output').get('bookings_summary') and \
                not new_state.get('booking_reference') and not new_state.get('restaurant_name') and \
                new_state.get('customer_email') and \
                len(new_state['tool_output']['bookings_summary']) == 1: # Only if single booking is found
            logger.info("Found booking ref and restaurant from tool output")
            bookings = new_state['tool_output']['bookings_summary']
            restaurants = new_state['tool_output'].get('booked_restaurants', []) # Handle potential missing restaurant info
            if bookings:
                # Assuming only one booking will match.  If multiple matches possible, handle the edge case better.
                new_state['booking_reference'] = bookings[0]['booking_reference']
                restaurant_id = bookings[0].get("restaurant_id")  # Attempt to get restaurant ID.
                if restaurant_id:
                  restaurant = next((r for r in restaurants if r['id'] == restaurant_id), None)
                else:
                  restaurant = None

                if restaurant:
                    new_state['restaurant_name'] = restaurant['name']
                elif restaurants:
                    new_state['restaurant_name'] = restaurants[0]['name'] #Fallback to the first restaurant if we didn't find a match.
                else:
                    logger.warning("Could not determine restaurant from summary.")

            # Clear missing info since we've filled it in now.
            new_state['missing_info'] = []

        # Check for missing required parameters for the determined tool
        target_tool_model = TOOL_NODE_MAP.get(new_state['current_intent'])

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


def confirm_booking_node(state: AgentState) -> AgentState:
    """
    Confirms booking details with the user, handles both existing and new bookings.
    """
    logger.info("Entering confirm_booking_node...")
    new_state = state.copy()
    user_message = state.get('user_message', '').strip().lower()
    original_intent = state.get('original_tool_intent')
    booking_summaries = state.get('tool_output', {}).get('bookings_summary', [])  # For existing bookings

    # ---- Handle "Cancel" or "No" ----
    if "cancel" in user_message or "no" in user_message:
        new_state['response'] = "Okay, cancelling the booking."
        new_state['current_intent'] = "respond_directly"  # Exit the flow.
        return new_state

    # ---- Handle New Booking Confirmation ----
    if original_intent == "CreateBookingTool":
        restaurant_name = state.get('restaurant_name')
        visit_date = state.get('VisitDate')
        visit_time = state.get('VisitTime')
        party_size = state.get('PartySize')
        # customer_email = state.get('customer_email')  # Or retrieve it.

        if not all([restaurant_name, visit_date, visit_time, party_size]):
            new_state['response'] = "I'm sorry, but I didn't get all the information. Please try again."
            new_state['current_intent'] = "ask_for_info"  # Go back and ask again
            new_state['missing_info'] = ["restaurant_name", "VisitDate", "VisitTime", "PartySize"]  # Reset missing info.
            return new_state

        #  Summarize the details for confirmation.
        confirmation_message = (
            f"OK, I'm about to book a table at {restaurant_name} for {party_size} people on {visit_date} at {visit_time}.  "
            "Is that correct?"
        )
        new_state['response'] = confirmation_message

        # If they confirm, transition to 'call_tool'
        if "yes" in user_message or "correct" in user_message or "ok" in user_message:
            new_state['current_intent'] = "call_tool"
        else:
            # If they say 'no', or don't confirm, let them start over.
            new_state['response'] = "Okay, let's start again."
            new_state['current_intent'] = "ask_for_info"
            new_state['missing_info'] = ['restaurant_name', 'VisitDate', 'VisitTime', 'PartySize']
            return new_state

    # ---- Handle Existing Booking (Selection) ----
    elif original_intent in ("UpdateBookingDetailsTool", "CancelBookingTool"): #Or, GetBookingDetailsTool
        # If in the confirmation state, the user should respond.
        try:
          selection_index = int(user_message) - 1
          if 0 <= selection_index < len(booking_summaries):
              selected_booking = booking_summaries[selection_index]
              new_state['booking_reference'] = selected_booking['booking_reference']
              new_state['restaurant_name'] = selected_booking.get('restaurant_name')
              new_state['missing_info'] = [] # we now have all the info
              new_state['current_intent'] = state['original_tool_intent'] # Go to the tool the user originally wanted
              new_state['response'] = "OK, I've selected that booking."
              logger.info(f"User selected booking: {selected_booking}")
          else:
              new_state['response'] = "Invalid selection. Please choose a number from the list."
              new_state['current_intent'] = "ask_for_info" #ask again.
              new_state['missing_info'] = ["booking_reference", "restaurant_name"]
        except ValueError:
            new_state['response'] = "Invalid input. Please enter the number of the booking or 'cancel'."
            new_state['current_intent'] = "ask_for_info" # ask again
            new_state['missing_info'] = ["booking_reference", "restaurant_name"]
    else: # Fallback if we get to this state unexpectedly.
        new_state['response'] = "I didn't understand. Please try again."
        new_state['current_intent'] = "ask_for_info"  # Go back to the start of the loop
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
        "column": "the column name to filter on (e.g., 'party_size', 'visit_date', 'status')",
        "value": "the value to filter by", "values": "the list of values to filter by",
        "lower": "the lower bound for the range", "upper": "the upper bound for the range",
        "pattern": "the search pattern (use % for wildcards)",
        # Add prompts for new query condition fields if they somehow become 'missing' directly
        "query_condition": "the query conditions (e.g., filters, limit, order by)",
        "booking_conditions": "the booking conditions (e.g., filters, limit, order by)",
        "restaurant_conditions": "the restaurant conditions (e.g., filters, limit, order by)",
    }
    # --- NEW LOGIC FOR CONFIRMATION STATE:  Modify prompts to be more specific ---
    if new_state.get('current_intent') == "confirm_booking":
        # If in the confirmation state, the user should respond.
        if new_state.get('original_tool_intent') == "CreateBookingTool":
            # No changes necessary, the confirm node handles the prompting.
            pass # The `confirm_booking` node handles prompting.
        else:
            #  Handle the existing bookings selection.
            new_state['response'] = "Please confirm which booking you want to modify, or say 'cancel'."
            return new_state


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
    target_tool_model = TOOL_NODE_MAP.get(tool_name)
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
            # Check if specific filters were applied to provide a more tailored response
            query_condition = state.get('query_condition')
            if query_condition and query_condition.filters:
                 response_message = "I found the following restaurants matching your criteria:\n"
            else:
                 response_message = "Here are the restaurants I know:\n"

            restaurants_info = [f"- {r['name']} ({r['address']})" for r in tool_output["data"]]
            response_message += "\n".join(restaurants_info)

        else: response_message = "No restaurants found matching your criteria."

    # Handle filter creation tools
    elif current_intent.startswith("Create") and current_intent.endswith("FilterTool"):
        if tool_output:
            filter_type = current_intent.replace("Create", "").replace("FilterTool", "").replace("Filter", "")
            response_message = f"Filter created successfully. You can now use this {filter_type.lower()} filter to search customer bookings with specific criteria."
        else:
            response_message = "I was unable to create the filter. Please check your parameters and try again."

    new_state['response'] = response_message
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
    workflow.add_node("confirm_booking", confirm_booking_node)

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
            "error": "prepare_response",         # If an error occurred in agent_node
            "confirm_booking": "confirm_booking", # if the user has multiple bookings to chose.
        },
    )

    # After a tool is executed, move to prepare the final response
    workflow.add_edge("tool_executor", "prepare_response")

    # After asking for info, the flow goes back to the agent to process the user's next input
    workflow.add_edge("ask_for_info", "agent")

    # Add the edge from confirm booking
    workflow.add_edge("confirm_booking", "agent")

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
    restaurant2 = Restaurant(name="The Great Bistro", address="456 Oak Ave", phone_number="555-5678")
    session.add_all([restaurant1, restaurant2])
    session.commit()
    session.refresh(restaurant1)
    session.refresh(restaurant2)

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

    # Add customers
    customer1 = Customer(
        first_name="John", surname="Doe", email="john.doe@example.com", mobile="1234567890",
        receive_email_marketing=True
    )
    customer2 = Customer(
        first_name="Alice", surname="Smith", email="alice.smith@example.com", mobile="0987654321",
        receive_email_marketing=False
    )
    session.add_all([customer1, customer2])
    session.commit()
    session.refresh(customer1)
    session.refresh(customer2)

    # Add existing bookings
    existing_booking1 = Booking(
        booking_reference="ABCDEFG",
        restaurant_id=restaurant1.id,
        customer_id=customer1.id,
        visit_date=date(2025, 12, 26),
        visit_time=time(19, 0, 0),
        party_size=4,
        channel_code="ONLINE",
        status="confirmed"
    )
    existing_booking2 = Booking(
        booking_reference="HIJKLMN",
        restaurant_id=restaurant2.id,
        customer_id=customer1.id,
        visit_date=date(2025, 11, 15),
        visit_time=time(18, 30, 0),
        party_size=2,
        channel_code="ONLINE",
        status="confirmed"
    )
    existing_booking3 = Booking(
        booking_reference="OPQRSTU",
        restaurant_id=restaurant1.id,
        customer_id=customer2.id,
        visit_date=date(2025, 10, 10),
        visit_time=time(20, 0, 0),
        party_size=5,
        channel_code="ONLINE",
        status="confirmed"
    )
    session.add_all([existing_booking1, existing_booking2, existing_booking3])
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
            query_condition=None, booking_conditions=None, restaurant_conditions=None,
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

    print("\n--- Conversation 8: Get Restaurants (Specific) Example - using filter now ---")
    run_conversation(app_graph, "Are there any restaurants with 'Fancy' in their name?", thread_id="get_restaurant_fancy_thread")

    print("\n--- Conversation 9: Get All Restaurants Example ---")
    run_conversation(app_graph, "List all restaurants.", thread_id="get_all_restaurants_thread_1")

    print("\n--- Conversation 10: Customer Bookings Summary with party size filter ---")
    run_conversation(app_graph, "What bookings does john.doe@example.com have for more than 3 people?", thread_id="customer_bookings_filter_thread_1")

    print("\n--- Conversation 11: Customer Bookings Summary with date range filter ---")
    run_conversation(app_graph, "What bookings does alice.smith@example.com have in October 2025?", thread_id="customer_bookings_filter_thread_2")


    # Clean up dummy data after tests
    session = next(get_db())
    session.query(Booking).delete()
    session.query(AvailabilitySlot).delete()
    session.query(Restaurant).delete()
    session.query(Customer).delete()
    session.query(CancellationReason).delete()
    session.commit()
    session.close()

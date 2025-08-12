# app/ai_chatbot.py
import functools, operator, re, inspect, os, dotenv
from typing import Dict, List, Any, Optional, Union, Tuple, Literal, TypedDict
from datetime import date, time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langchain.chat_models.base import init_chat_model
from langchain_core.runnables.history import BaseChatMessageHistory
from langgraph.checkpoint.sqlite import SqliteSaver

# Import your existing AI tools interface
from app.ai_tools import AIToolCallingInterface

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = dotenv.get_key(".env", "gemini_api")
model = init_chat_model(model_name="google_genai:gemini-1.5-flash-001").bind_tools(
    list(AIToolCallingInterface.__dict__.values()) # Dynamically get all callable tools
)

# Define the AgentState - using 'messages' for LangGraph's checkpointer
class AgentState(TypedDict):
    messages: List[BaseMessage] # This will store the chat history
    current_task: Optional[str] # e.g., 'list_restaurants', 'search_availability', 'create_booking', 'find_customer_bookings', 'update_booking_details', 'cancel_booking'
    sub_task_state: Optional[str] # For multi-step tasks: 'awaiting_selection', 'awaiting_details', 'awaiting_cancellation_reason', 'awaiting_confirmation'
    tool_parameters: Dict[str, Any] # Parameters collected for the current task
    customer_email: Optional[str] # Store customer email for repeated use
    bookings_found: Optional[List[Dict[str, Any]]] # Store list of bookings found for selection
    selected_booking_ref: Optional[str] # The reference of the booking chosen for update/cancel
    selected_restaurant_name: Optional[str] # The restaurant_name chosen for update/cancel
    booking_to_modify: Optional[Dict[str, Any]] # Stores the entire booking object chosen for update/cancel

# --- Helper Classes and Functions ---
class HTMLTableCreator:
    prefix = "<table><thead>"
    postfix = "</tbody></table>"
    def __init__(self, columns=None):
        self.thead = "</thead><tbody>"
        if columns:
            self.thead = "<tr>" + "".join("<th>"+col+"</th>" for col in columns) + "</tr>" + self.thead
        self.tbody = ""
    def add_row(self, row_data):
        tr = "<tr>"
        for entry in row_data:
            tr += "<td>"+str(entry)+"</td>"
        tr += "</tr>"
        self.tbody += tr
    def from_dict(self, content:dict):
        # Creates a two-column table from a dictionary (Key | Value)
        for key, value in content.items():
            self.tbody += f"<tr><th>{key}</th><td>{value}</td></tr>"
        return self.html()
    def html(self):
        return HTMLTableCreator.prefix + self.thead + self.tbody + HTMLTableCreator.postfix

# Helper function to get missing parameters for a tool
def get_missing_parameters(tool_name: str, collected_params: Dict[str, Any]) -> List[str]:
    # Find the tool's original function to inspect its signature
    # Ensure to only include actual tools bound to the model
    bound_tools = model.tools
    func = next((t.func for t in bound_tools if t.name == tool_name), None)
    if not func:
        return []

    sig = inspect.signature(func)
    missing_params = []
    for param_name, param_info in sig.parameters.items():
        if param_name == 'self':
            continue
        
        # Check if the parameter is required (no default value) and not in collected_params
        if param_info.default == inspect.Parameter.empty and param_name not in collected_params:
            # create_booking has many optional customer details which are not strictly required for the *tool* to be called,
            # but might be for the *API*. We let the LLM handle asking for these.
            # Only flag truly missing *required* parameters for the tool call itself.
            if tool_name == "create_booking" and param_name.startswith("customer_") and param_name not in ["customer_email", "customer_first_name", "customer_surname", "customer_mobile"]:
                 continue # These are often optional at the tool level or gathered iteratively.

            missing_params.append(param_name)
    return missing_params

# Helper for confirmation
def is_confirmation(text: str) -> bool:
    text = text.lower()
    confirmation_keywords = ["yes", "yep", "confirm", "proceed", "go ahead", "ok", "no problem", "correct", "right", "true", "positive"]
    return any(keyword in text for keyword in confirmation_keywords)

def is_negation(text: str) -> bool:
    text = text.lower()
    negation_keywords = ["no", "nope","not", "don't", "cancel", "stop", "wrong", "false", "negative", "nevermind"]
    return any(keyword in text for keyword in negation_keywords)

# --- LangGraph Nodes ---

# The main agent node logic
def agent_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    current_task = state.get("current_task")
    sub_task_state = state.get("sub_task_state")
    tool_parameters = state.get("tool_parameters", {})
    bookings_found = state.get("bookings_found")
    # Get last human message
    user_input = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    selected_booking_ref = state.get("selected_booking_ref")
    selected_restaurant_name = state.get("selected_restaurant_name")

    # --- Handle Confirmation State ---
    if sub_task_state == "awaiting_confirmation":
        if is_confirmation(user_input):
            messages.append(AIMessage(content="<p>Acknowledged. Proceeding with the action...</p>"))
            state["sub_task_state"] = None # Ready to proceed to call_tool
            return state
        elif is_negation(user_input):
            messages.append(AIMessage(content="<p>Action cancelled. What else can I help you with?</p>"))
            # Reset the state to return to initial stage
            state["current_task"] = None
            state["sub_task_state"] = None
            state["tool_parameters"] = {}
            state["bookings_found"] = None
            state["selected_booking_ref"] = None
            state["selected_restaurant_name"] = None
            state["booking_to_modify"] = None
            return state
        else:
            messages.append(AIMessage(content="<p>Please confirm with 'yes' or 'no'.</p>"))
            return state

    # --- Logic for handling booking selection (for update/cancel) ---
    if current_task in ["update_booking_details", "cancel_booking"] and sub_task_state == "awaiting_selection":
        if bookings_found:
            num_match = re.search(r"\b\d+\b", user_input) # ensure it's a whole number
            if num_match:
                try:
                    selected_index = int(num_match.group()) - 1
                    if 0 <= selected_index < len(bookings_found):
                        selected_booking = bookings_found[selected_index]
                        state["selected_booking_ref"] = selected_booking["booking_reference"]
                        state["selected_restaurant_name"] = selected_booking["restaurant_name"]
                        tool_parameters["restaurant_name"] = selected_booking["restaurant_name"]
                        tool_parameters["booking_reference"] = selected_booking["booking_reference"]
                        state["booking_to_modify"] = selected_booking
                        
                        if current_task == "cancel_booking":
                            state["sub_task_state"] = "awaiting_cancellation_reason"
                            # Use AIToolCallingInterface directly since it's an internal call
                            cancellation_reasons = AIToolCallingInterface.list_cancellation_reasons()
                            # Format cancellation reasons nicely
                            columns = ["Reason ID", "Title", "Description"]
                            reason_html = HTMLTableCreator(columns)
                            for r in cancellation_reasons:
                                reason_html.add_row([r.get("id", "N/A"), r.get("title", "N/A"), r.get("description", "N/A")])
                            
                            messages.append(AIMessage(content=f"<p>You've selected booking <b>{selected_booking['booking_reference']}</b> for <i>{selected_booking['restaurant_name']}</i>. Please provide the ID of the cancellation reason from the following table: </p>"+reason_html.html()))
                        elif current_task == "update_booking_details":
                            state["sub_task_state"] = "awaiting_details"
                            messages.append(AIMessage(content=f"<p>You've selected booking <b>{selected_booking['booking_reference']}</b> for <i>{selected_booking['restaurant_name']}</i>. What details would you like to update (e.g., VisitDate, VisitTime, PartySize, SpecialRequests)?</p>"))
                        return state
                    else:
                        messages.append(AIMessage(content=f"<p>That booking number is out of range. Please select a number between 1 and {len(bookings_found)}.</p>"))
                        return state
                except ValueError:
                    messages.append(AIMessage(content="<p>I couldn't identify the booking you wish to modify. Please specify the booking number as an integer value (1, 2, 3...).</p>"))
                    return state
            else:
                messages.append(AIMessage(content="<p>I couldn't identify the booking you wish to modify. Please specify the booking number as an integer value (1, 2, 3...).</p>"))
                return state

    # --- Logic for collecting cancellation reason ID ---
    if current_task == "cancel_booking" and sub_task_state == "awaiting_cancellation_reason":
        try:
            reason_id = int(re.search(r'\b\d+', user_input).group())
            tool_parameters["cancellationReasonId"] = reason_id
            tool_parameters["micrositeName"] = selected_restaurant_name # micrositeName usually same as restaurant_name
            state["tool_parameters"] = tool_parameters
            state["sub_task_state"] = "awaiting_confirmation" # Now ask for confirmation
            messages.append(AIMessage(content=f"<p>Are you sure you want to cancel booking <b>{selected_booking_ref}</b> for <i>{selected_restaurant_name}</i> with reason No.<b>{reason_id}</b>? Please confirm.</p>"))
            return state
        except (AttributeError, ValueError):
            messages.append(AIMessage(content="<p>Please provide a valid cancellation reason ID (a number).</p>"))
            return state

    # --- Logic for collecting update details ---
    if current_task == "update_booking_details" and sub_task_state == "awaiting_details":
        # Let the model try to extract update parameters from the latest user input
        # We invoke the model to get a potential tool_call for update_booking_details
        response = model.invoke({"input": user_input, "messages": messages, "agent_scratchpad": []})
        
        if response.tool_calls and response.tool_calls[0].name == current_task:
            new_params = response.tool_calls[0].args
            
            # Filter out restaurant_name and booking_reference if they came from the model, as they are already set.
            for old_param in ["restaurant_name", "booking_reference"]:
                if old_param in new_params:
                    new_params.pop(old_param)

            if new_params: # Only update if new parameters were actually extracted
                tool_parameters.update(new_params)
                state["tool_parameters"] = tool_parameters
                messages.append(response) # Add the new tool_call message with updated params
                
                # Summarize the changes for confirmation
                messages.append(AIMessage(content=f"<p>Confirm you wish to update booking <b>{selected_booking_ref}</b> for <i>{selected_restaurant_name}</i> with the following changes: </p>{HTMLTableCreator().from_dict(new_params)}"))
                state["sub_task_state"] = "awaiting_confirmation"
                return state
            else:
                messages.append(AIMessage(content="<p>I couldn't detect any specific updates. What exactly would you like to update? Please specify (e.g., 'change date to 2025-01-01', 'party size to 5').</p>"))
                return state
        else:
            messages.append(AIMessage(content="<p>What exactly would you like to update? Please specify (e.g., 'change date to 2025-01-01', 'party size to 5').</p>"))
            return state

    # --- Initial intent detection and then parameter collection for new tasks ---
    if current_task is None or (current_task not in ["update_booking_details", "cancel_booking"] and not tool_parameters):
        # Pass the entire messages history for context
        response = model.invoke({"input": user_input, "messages": messages, "agent_scratchpad": []})
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.name
            params = tool_call.args
            
            state["current_task"] = tool_name
            state["tool_parameters"] = params
            messages.append(response) # Add the AI's tool_call message to history
            
            missing = get_missing_parameters(tool_name, params)
            if missing:
                columns = ["Parameter"]
                missing_html = HTMLTableCreator(columns)
                for m in missing:
                    missing_html.add_row([m])
                messages.append(AIMessage(content=f"<p>I need more information to <i>{tool_name}</i>. Could you please provide the following:</p> {missing_html.html()}"))
                return state
            else:
                # If all parameters are there, for non-update/cancel tasks, ready to proceed.
                return state # Ready for call_tool
        else:
            # Model is just chatting or clarifying
            messages.append(response)
            return state

    # If current_task is set, and we are not in special sub-states,
    # try to collect remaining parameters from user input.
    if current_task is not None and sub_task_state is None:
        response = model.invoke({"input": user_input, "messages": messages, "agent_scratchpad": []})
        
        if response.tool_calls and response.tool_calls[0].name == current_task:
            tool_parameters.update(response.tool_calls[0].args) # Merge new parameters
            state["tool_parameters"] = tool_parameters
            messages.append(response) # Add the AI's tool_call message
            
            missing = get_missing_parameters(current_task, tool_parameters)
            if missing:
                columns = ["Parameter"]
                missing_html = HTMLTableCreator(columns)
                for m in missing:
                    missing_html.add_row([m])
                messages.append(AIMessage(content=f"<p>I still need more information for your <i>{current_task}</i> request. Please provide:</p>{missing_html.html()}"))
                return state
            else:
                # If all parameters are now collected, and it's not update/cancel, ready to call tool.
                return state 
        else:
            # Model didn't complete the tool call, just replied naturally.
            messages.append(response)
            return state

    return state # Should ideally not be reached if logic covers all cases

# Node for executing the tool call
def call_tool(state: AgentState) -> Dict[str, Any]:
    current_task = state["current_task"]
    tool_parameters = state["tool_parameters"]
    messages = state["messages"]

    # Retrieve the actual tool function from AIToolCallingInterface directly
    selected_tool_func = getattr(AIToolCallingInterface, current_task, None)
    tool_output = None
    if not selected_tool_func:
        tool_output = f"Error: Tool '{current_task}' not found in AIToolCallingInterface."
    else:
        try:
            # Convert date/time strings to objects if required by the tool function
            # Create a copy of parameters to modify for function call if needed
            invoke_params = tool_parameters.copy() 
            for param, value in invoke_params.items():
                if param == "VisitDate" and isinstance(value, str):
                    try:
                        invoke_params[param] = date.fromisoformat(value)
                    except ValueError:
                        pass # Keep as string if conversion fails, let tool handle it
                elif param == "VisitTime" and isinstance(value, str):
                    try:
                        invoke_params[param] = time.fromisoformat(value)
                    except ValueError:
                        pass # Keep as string if conversion fails, let tool handle it

            tool_output = selected_tool_func(**invoke_params)
            
            if isinstance(tool_output, dict) and "error" in tool_output:
                tool_output = f"Error: {tool_output['error']}"
            elif isinstance(tool_output, list) and not tool_output: # Handle empty list results for some tools
                tool_output = "No results found."

        except Exception as e:
            tool_output = f"Error executing tool '{current_task}': {str(e)}"
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    # Find the tool_call_id to associate the ToolMessage correctly
    tool_call_id = None
    # Iterate backwards through messages to find the most recent AIMessage with tool_calls
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Find the tool call that matches the current_task if possible, or just the latest one
            for tc in msg.tool_calls:
                if tc.name == current_task:
                    tool_call_id = tc.id
                    break
            if tool_call_id:
                break
    
    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id))

    # Special handling for tool outputs for user display
    if current_task == "find_customer_bookings" and not isinstance(tool_output, str) and "error" not in str(tool_output) and tool_output != "No results found.":
        bookings = tool_output
        state["bookings_found"] = bookings
        if bookings:
            # Dynamically get columns from the first booking if available
            columns = ["#"] + [key.replace("_", " ").title() for key in bookings[0].keys()] if bookings else ["#", "Booking Details"]
            bookings_html = HTMLTableCreator(columns)
            for i, booking in enumerate(bookings):
                row_data = [i+1] + [booking.get(key, "N/A") for key in bookings[0].keys()] # Ensure order matches columns
                bookings_html.add_row(row_data)
            
            # Check if this find_customer_bookings was a precursor to update/cancel
            original_task_after_find = state["tool_parameters"].pop("original_task_after_find_bookings", None)
            if original_task_after_find in ["update_booking_details", "cancel_booking"]:
                state["current_task"] = original_task_after_find
                state["sub_task_state"] = "awaiting_selection"
                messages.append(AIMessage(content="<p>I found the following bookings for you:</p>" + bookings_html.html() + "<p>Please tell me which booking you'd like to update/cancel by its number (e.g., '1' for the first one).</p>"))
                return state
            
            messages.append(AIMessage(content="<p>Here are your bookings:</p>" + bookings_html.html()))
        else:
            messages.append(AIMessage(content="<p>No bookings found for the provided criteria.</p>"))
    elif current_task == "list_restaurants" and isinstance(tool_output, list) and not isinstance(tool_output, str):
        if tool_output:
            # Dynamically get columns from the first restaurant if available
            columns = [key.replace("restaurant_", "").replace("_", " ").title() for key in tool_output[0].keys()] if tool_output else ["Restaurant Details"]
            restaurants_html = HTMLTableCreator(columns)
            for restaurant in tool_output:
                row_data = [restaurant.get(key, 'N/A') for key in tool_output[0].keys()] # Ensure order matches columns
                restaurants_html.add_row(row_data)
            messages.append(AIMessage(content="<p>Here are the restaurants:</p>" + restaurants_html.html()))
        else:
            messages.append(AIMessage(content="<p>No restaurants found.</p>"))
    elif current_task == "search_availability" and isinstance(tool_output, dict) and "results" in tool_output:
        availabilities = tool_output["results"]
        if availabilities:
            columns = [key.replace("_", " ").title() for key in availabilities[0].keys()] if availabilities else ["Availability Details"]
            availability_html = HTMLTableCreator(columns)
            for slot in availabilities:
                row_data = [slot.get(key, "N/A") for key in availabilities[0].keys()]
                availability_html.add_row(row_data)
            messages.append(AIMessage(content="<p>Here are the available slots:</p>" + availability_html.html()))
        else:
            messages.append(AIMessage(content="<p>No availability found for the given criteria.</p>"))
    elif current_task == "create_booking" and isinstance(tool_output, dict) and "booking_reference" in tool_output:
        booking_ref = tool_output.get("booking_reference", "N/A")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking successfully created for <i>{restaurant_name}</i> with reference <b>{booking_ref}</b>.</p>"))
    elif current_task == "cancel_booking" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        booking_ref = tool_parameters.get("booking_reference", "the booking")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking <b>{booking_ref}</b> for <i>{restaurant_name}</i> has been successfully cancelled.</p>"))
    elif current_task == "update_booking_details" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        booking_ref = tool_parameters.get("booking_reference", "the booking")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking <b>{booking_ref}</b> for <i>{restaurant_name}</i> has been successfully updated.</p>"))
    else:
        # For all other tool calls, provide a general confirmation or error message
        messages.append(AIMessage(content=f"<p>The {current_task} operation completed with result: {tool_output}</p>"))

    # Reset state to return to initial stage after a task is completed (or for non-multi-step tasks)
    if state.get("sub_task_state") not in ["awaiting_selection", "awaiting_details", "awaiting_cancellation_reason", "awaiting_confirmation"]:
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None

    return state

# --- LangGraph Workflow Definition ---
# Configure SQLite checkpointer
memory = SqliteSaver.from_conn_string(":memory:") # Use in-memory for demonstration

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("call_tool", call_tool)

# Define entry point
workflow.add_edge(START, "agent")

# Define conditional transitions from the agent node
def route_agent_action(state: AgentState):
    """
    Decides whether the agent should call a tool, continue collecting information,
    or has finished a sub-task.
    """
    # If the agent is awaiting confirmation, it stays in the agent node
    if state.get("sub_task_state") == "awaiting_confirmation":
        return "agent"

    # If the agent generated a tool call and all parameters are collected (sub_task_state is None)
    # This specifically means it passed the confirmation step for update/cancel, or it's a direct tool call.
    if state["messages"] and state["messages"][-1].tool_calls:
        # If it's an update/cancel task, ensure it's not in a sub-state and has all params.
        # This implicitly means it passed the confirmation if it was required.
        if state["current_task"] in ["update_booking_details", "cancel_booking"] and state.get("sub_task_state") is None:
            return "call_tool"
        
        # For other direct tool calls, if no missing params and no sub-state, proceed.
        missing = get_missing_parameters(state["current_task"], state["tool_parameters"])
        if not missing and state.get("sub_task_state") is None:
            return "call_tool"
        
    # If the agent needs more information (missing parameters, or in a sub-task like awaiting selection/details/reason)
    # it means it needs more user input or processing within the agent node itself.
    # Note: awaiting_confirmation is handled first, so this catches other sub-tasks.
    if state.get("sub_task_state") is not None or get_missing_parameters(state.get("current_task", ""), state.get("tool_parameters", {})):
        return "agent"
    
    # Default: loop back to agent for further processing or if no clear action yet
    return "agent"

workflow.add_conditional_edges(
    "agent",
    route_agent_action,
    {
        "call_tool": "call_tool",
        "agent": "agent", # Loop back to agent for more input or internal processing
    },
)

# After a tool is called, always return to the agent node to process the output
# and determine the next step (which might be to reset or continue a sub-task).
workflow.add_edge("call_tool", "agent")

# Compile the graph with the checkpointer

app = workflow.compile(checkpointer=memory)

# Define initial state for new conversations
initial_agent_state = {
    "messages": [AIMessage(content="Hello! I'm your restaurant booking assistant. I can help you check availability, make bookings, or manage existing reservations. What would you like to do?")],
    "current_task": None,
    "sub_task_state": None,
    "tool_parameters": {},
    "customer_email": None,
    "bookings_found": None,
    "selected_booking_ref": None,
    "selected_restaurant_name": None,
    "booking_to_modify": None
}
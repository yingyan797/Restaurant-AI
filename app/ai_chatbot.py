import functools, operator, re, inspect, os, dotenv
from typing import Dict, List, Any, Optional, Union, Tuple, Literal, TypedDict
from datetime import date, time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools.structured import BaseTool, StructuredTool
from langgraph.graph import StateGraph, END, START
from langchain.chat_models.base import init_chat_model
from langchain_core.runnables.history import BaseChatMessageHistory
from langgraph.checkpoint.sqlite import SqliteSaver

# Import your existing AI tools interface
from app.ai_tools import AIToolCallingInterface

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = dotenv.get_key(".env","gemini_api") # Ensure this matches your .env key name

# Collect only actual Tool instances
all_ai_tools = list[StructuredTool]()
logging.debug("Here are the tools found for this task:")
for key in dir(AIToolCallingInterface):
    tool = getattr(AIToolCallingInterface, key)
    if isinstance(tool, StructuredTool):
        all_ai_tools.append(tool)
        print(tool.name, type(tool.func))
    

# Define the ChatPromptTemplate to provide system instructions
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and efficient restaurant booking assistant. Your primary goal is to assist users with restaurant-related queries, including:
- Listing available restaurants.
- Searching for booking availability.
- Creating new bookings.
- Finding, updating, or canceling customer's existing bookings.

When responding, use the available tools to fulfill user requests. Here's how:

1. Prioritize Tool Usage: If a user's request can be directly addressed by one of your tools, propose or call that tool.
2. Conversational Fallback: If a request does not fit any tool, respond conversationally and offer alternative help."""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Corrected model initialization with prompt chaining
model = init_chat_model(
    model="gemini-2.0-flash-lite",
    model_provider="google_genai"
)

# Chain the prompt with the model and bind tools
# This ensures the model receives the system instructions before processing messages
model_with_tools = model.bind_tools(all_ai_tools)
chain = prompt | model_with_tools


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
    logging.debug(f"Checking missing parameters for tool: {tool_name} with collected: {collected_params}")
    func = next((t.func for t in all_ai_tools if t.name == tool_name), None)
    if not func:
        logging.warning(f"Tool function not found for '{tool_name}'.")
        return []

    sig = inspect.signature(func)
    missing_params = []
    for param_name, param_info in sig.parameters.items():
        if param_name == 'self':
            continue
        
        if param_info.default == inspect.Parameter.empty and param_name not in collected_params:
            if tool_name == "create_booking" and param_name.startswith("customer_") and param_name not in ["customer_email", "customer_first_name", "customer_surname", "customer_mobile"]:
                continue

            missing_params.append(param_name)
    logging.debug(f"Missing parameters for {tool_name}: {missing_params}")
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
    logging.info(f"--- Entering agent_node ---")
    logging.debug(f"AgentState on entry: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}, tool_parameters={state.get('tool_parameters')}")

    messages = state.get("messages", [])
    current_task = state.get("current_task")
    sub_task_state = state.get("sub_task_state")
    tool_parameters = state.get("tool_parameters", {})
    bookings_found = state.get("bookings_found")
    user_input = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    selected_booking_ref = state.get("selected_booking_ref")
    selected_restaurant_name = state.get("selected_restaurant_name")

    # --- Handle Confirmation State ---
    if sub_task_state == "awaiting_confirmation":
        logging.info(f"Agent is in 'awaiting_confirmation' state. User input: '{user_input}'")
        if is_confirmation(user_input):
            messages.append(AIMessage(content="<p>Acknowledged. Proceeding with the action...</p>"))
            state["sub_task_state"] = None
            logging.info("Confirmation received. Setting sub_task_state to None.")
            return state
        elif is_negation(user_input):
            messages.append(AIMessage(content="<p>Action cancelled. What else can I help you with?</p>"))
            logging.info("Negation received. Resetting all task states.")
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
            logging.info("Invalid confirmation input. Asking user to confirm again.")
            return state

    # --- Logic for handling booking selection (for update/cancel) ---
    if current_task in ["update_booking_details", "cancel_booking"] and sub_task_state == "awaiting_selection":
        logging.info(f"Agent is in 'awaiting_selection' state for {current_task}. User input: '{user_input}'")
        if bookings_found:
            num_match = re.search(r"\b\d+\b", user_input)
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
                            cancellation_reasons = AIToolCallingInterface.list_cancellation_reasons()
                            columns = ["Reason ID", "Title", "Description"]
                            reason_html = HTMLTableCreator(columns)
                            for r in cancellation_reasons:
                                reason_html.add_row([r.get("id", "N/A"), r.get("title", "N/A"), r.get("description", "N/A")])
                            
                            messages.append(AIMessage(content=f"<p>You've selected booking <b>{selected_booking['booking_reference']}</b> for <i>{selected_booking['restaurant_name']}</i>. Please provide the ID of the cancellation reason from the following table: </p>"+reason_html.html()))
                            logging.info(f"Selected booking {selected_booking['booking_reference']} for cancellation. Transitioning to 'awaiting_cancellation_reason'.")
                        elif current_task == "update_booking_details":
                            state["sub_task_state"] = "awaiting_details"
                            messages.append(AIMessage(content=f"<p>You've selected booking <b>{selected_booking['booking_reference']}</b> for <i>{selected_booking['restaurant_name']}</i>. What details would you like to update (e.g., VisitDate, VisitTime, PartySize, SpecialRequests)?</p>"))
                            logging.info(f"Selected booking {selected_booking['booking_reference']} for update. Transitioning to 'awaiting_details'.")
                        return state
                    else:
                        messages.append(AIMessage(content=f"<p>That booking number is out of range. Please select a number between 1 and {len(bookings_found)}.</p>"))
                        logging.info("Invalid booking number selected (out of range).")
                        return state
                except ValueError:
                    messages.append(AIMessage(content="<p>I couldn't identify the booking you wish to modify. Please specify the booking number as an integer value (1, 2, 3...).</p>"))
                    logging.info("Invalid booking number input (not an integer).")
                    return state
            else:
                messages.append(AIMessage(content="<p>I couldn't identify the booking you wish to modify. Please specify the booking number as an integer value (1, 2, 3...).</p>"))
                logging.info("No booking number identified in user input.")
                return state

    # --- Logic for collecting cancellation reason ID ---
    if current_task == "cancel_booking" and sub_task_state == "awaiting_cancellation_reason":
        logging.info(f"Agent in 'awaiting_cancellation_reason'. User input: '{user_input}'")
        try:
            reason_id_match = re.search(r'\b\d+', user_input)
            if reason_id_match:
                reason_id = int(reason_id_match.group())
                tool_parameters["cancellationReasonId"] = reason_id
                tool_parameters["micrositeName"] = selected_restaurant_name
                state["tool_parameters"] = tool_parameters
                state["sub_task_state"] = "awaiting_confirmation"
                messages.append(AIMessage(content=f"<p>Are you sure you want to cancel booking <b>{selected_booking_ref}</b> for <i>{selected_restaurant_name}</i> with reason No.<b>{reason_id}</b>? Please confirm.</p>"))
                logging.info(f"Cancellation reason ID '{reason_id}' collected. Transitioning to 'awaiting_confirmation'.")
                return state
            else:
                messages.append(AIMessage(content="<p>Please provide a valid cancellation reason ID (a number).</p>"))
                logging.info("No valid cancellation reason ID found in input.")
                return state
        except (AttributeError, ValueError):
            messages.append(AIMessage(content="<p>Please provide a valid cancellation reason ID (a number).</p>"))
            logging.info("Error parsing cancellation reason ID.")
            return state

    # --- Logic for collecting update details ---
    if current_task == "update_booking_details" and sub_task_state == "awaiting_details":
        logging.info(f"Agent in 'awaiting_details' for update_booking_details. User input: '{user_input}')")
        response = chain.invoke({"messages": messages}) # Use the new chain here
        
        if response.tool_calls and response.tool_calls[0]['name'] == current_task:
            new_params = response.tool_calls[0]['args']
            
            for old_param in ["restaurant_name", "booking_reference"]:
                if old_param in new_params:
                    new_params.pop(old_param)

            if new_params:
                tool_parameters.update(new_params)
                state["tool_parameters"] = tool_parameters
                messages.append(response)
                
                messages.append(AIMessage(content=f"<p>Confirm you wish to update booking <b>{selected_booking_ref}</b> for <i>{selected_restaurant_name}</i> with the following changes: </p>{HTMLTableCreator().from_dict(new_params)}"))
                state["sub_task_state"] = "awaiting_confirmation"
                logging.info(f"Update parameters extracted: {new_params}. Transitioning to 'awaiting_confirmation'.")
                return state
            else:
                messages.append(AIMessage(content="<p>I couldn't detect any specific updates. What exactly would you like to update? Please specify (e.g., 'change date to 2025-01-01', 'party size to 5').</p>"))
                logging.info("Model did not extract new update parameters.")
                return state
        else:
            messages.append(AIMessage(content="<p>What exactly would you like to update? Please specify (e.g., 'change date to 2025-01-01', 'party size to 5').</p>"))
            logging.info("Model did not suggest tool call for update details or suggested wrong tool.")
            return state

    # --- Initial intent detection and then parameter collection for new tasks ---
    if current_task is None or (current_task not in ["update_booking_details", "cancel_booking"] and not tool_parameters):
        logging.info("No current task or initial task detection. Invoking model for intent.")
        # Include customer email in context
        context = {"messages": messages}
        if state.get("customer_email"):
            enhanced_messages = [AIMessage(content=f"Customer email available: {state['customer_email']}")]
            enhanced_messages.extend(messages)
            context["messages"] = enhanced_messages
        else:
            raise ValueError("No email found")
        response = chain.invoke(context)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call['name']
            params = tool_call['args']
            logging.info(f"Model suggested tool call: {tool_name} with params: {params}")
            
            state["current_task"] = tool_name
            state["tool_parameters"] = params
            messages.append(response)
            
            missing = get_missing_parameters(tool_name, params)
            if missing:
                columns = ["Parameter"]
                missing_html = HTMLTableCreator(columns)
                for m in missing:
                    missing_html.add_row([m])
                messages.append(AIMessage(content=f"<p>I need more information to <i>{tool_name}</i>. Could you please provide the following:</p> {missing_html.html()}"))
                logging.info(f"Missing parameters found: {missing}. Staying in agent to collect.")
                return state
            else:
                logging.info("All parameters collected for new task. Ready for tool call.")
                return state
        else:
            messages.append(response)
            logging.info("Model did not suggest a tool call. Conversational response.")
            return state

    # If current_task is set, and we are not in special sub-states (like awaiting_confirmation),
    # try to collect remaining parameters from user input for the ongoing task.
    if current_task is not None and sub_task_state is None:
        logging.info(f"Current task {current_task} active, no sub-state. Invoking model for parameter collection.")
        response = chain.invoke({"messages": messages}) # Use the new chain here
        
        if response.tool_calls and response.tool_calls[0]['name'] == current_task:
            new_params = response.tool_calls[0]['args']
            if new_params:
                tool_parameters.update(new_params) 
                state["tool_parameters"] = tool_parameters
            messages.append(response)
            
            missing = get_missing_parameters(current_task, tool_parameters)
            if missing:
                columns = ["Parameter"]
                missing_html = HTMLTableCreator(columns)
                for m in missing:
                    missing_html.add_row([m])
                messages.append(AIMessage(content=f"<p>I still need more information for your <i>{current_task}</i> request. Please provide:</p>{missing_html.html()}"))
                logging.info(f"Still missing parameters for {current_task}: {missing}. Staying in agent.")
                return state
            else:
                logging.info(f"All parameters collected for {current_task}. Ready for tool call.")
                return state
        else:
            messages.append(response)
            logging.info(f"Model didn't complete tool call for {current_task}, replied conversationally. Staying in agent.")
            return state

    logging.warning("Agent node reached end without explicit return. This should not happen frequently.")
    return state

# Node for executing the tool call
def call_tool(state: AgentState) -> Dict[str, Any]:
    logging.info(f"--- Entering call_tool node ---")
    logging.debug(f"State on entry: current_task={state.get('current_task')}, tool_parameters={state.get('tool_parameters')}")
    current_task = state["current_task"]
    tool_parameters = state["tool_parameters"]
    messages = state["messages"]

    selected_tool = next((t for t in all_ai_tools if t.name == current_task), None)
    selected_tool_func = selected_tool.func if selected_tool else None

    tool_output = None
    if not selected_tool_func:
        tool_output = f"Error: Tool '{current_task}' not found in AIToolCallingInterface."
        logging.error(f"Tool function not found: {current_task}")
    else:
        try:
            invoke_params = tool_parameters.copy() 
            for param, value in invoke_params.items():
                if param == "VisitDate" and isinstance(value, str):
                    try:
                        invoke_params[param] = date.fromisoformat(value)
                    except ValueError:
                        logging.warning(f"Failed to convert VisitDate '{value}' to date object.")
                        pass
                elif param == "VisitTime" and isinstance(value, str):
                    try:
                        invoke_params[param] = time.fromisoformat(value)
                    except ValueError:
                        logging.warning(f"Failed to convert VisitTime '{value}' to time object.")
                        pass

            logging.info(f"Calling tool '{current_task}' with parameters: {invoke_params}")
            tool_output = selected_tool_func(**invoke_params)
            
            if isinstance(tool_output, dict) and "error" in tool_output:
                tool_output = f"Error: {tool_output['error']}"
                logging.error(f"Tool '{current_task}' returned an error: {tool_output}")
            elif isinstance(tool_output, list) and not tool_output:
                tool_output = "No results found."
                logging.info(f"Tool '{current_task}' returned no results.")

        except Exception as e:
            tool_output = f"Error executing tool '{current_task}': {str(e)}"
            logging.exception(f"Exception during tool execution for '{current_task}'")

    tool_call_id = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['name'] == current_task:
                    tool_call_id = tc['id']
                    break
            if tool_call_id:
                break
    
    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id))
    logging.debug(f"ToolMessage added: {str(tool_output)[:100]}...")

    # --- Post-tool-call state management and response generation ---
    # This section determines what message to send to the user and how to update state
    # BEFORE routing to the next node.

    if current_task == "find_customer_bookings" and not isinstance(tool_output, str) and "error" not in str(tool_output) and tool_output != "No results found.":
        bookings = tool_output
        state["bookings_found"] = bookings
        if bookings:
            columns = ["#"] + [key.replace("_", " ").title() for key in bookings[0].keys()] if bookings else ["#", "Booking Details"]
            bookings_html = HTMLTableCreator(columns)
            for i, booking in enumerate(bookings):
                row_data = [i+1] + [booking.get(key, "N/A") for key in bookings[0].keys()]
                bookings_html.add_row(row_data)
            
            original_task_after_find = state["tool_parameters"].pop("original_task_after_find_bookings", None)
            if original_task_after_find in ["update_booking_details", "cancel_booking"]:
                state["current_task"] = original_task_after_find
                state["sub_task_state"] = "awaiting_selection"
                messages.append(AIMessage(content="<p>I found the following bookings for you:</p>" + bookings_html.html() + "<p>Please tell me which booking you'd like to update/cancel by its number (e.g., '1' for the first one).</p>"))
                logging.info(f"Find customer bookings: found {len(bookings)} bookings. Transitioning to '{original_task_after_find}' and 'awaiting_selection'.")
            else:
                messages.append(AIMessage(content="<p>Here are your bookings:</p>" + bookings_html.html()))
                state["current_task"] = None
                state["tool_parameters"] = {}
                state["sub_task_state"] = None
                state["bookings_found"] = None
                state["selected_booking_ref"] = None
                state["selected_restaurant_name"] = None
                state["booking_to_modify"] = None
                logging.info("Find customer bookings: found bookings. Task completed, state reset.")

        else:
            messages.append(AIMessage(content="<p>No bookings found for the provided criteria.</p>"))
            state["current_task"] = None
            state["tool_parameters"] = {}
            state["sub_task_state"] = None
            state["bookings_found"] = None
            state["selected_booking_ref"] = None
            state["selected_restaurant_name"] = None
            state["booking_to_modify"] = None
            logging.info("Find customer bookings: No bookings found. Task completed, state reset.")

    elif current_task == "list_restaurants" and isinstance(tool_output, list) and not isinstance(tool_output, str):
        if tool_output:
            columns = [key.replace("restaurant_", "").replace("_", " ").title() for key in tool_output[0].keys()] if tool_output else ["Restaurant Details"]
            restaurants_html = HTMLTableCreator(columns)
            for restaurant in tool_output:
                row_data = [restaurant.get(key, 'N/A') for key in tool_output[0].keys()]
                restaurants_html.add_row(row_data)
            messages.append(AIMessage(content="<p>Here are the restaurants:</p>" + restaurants_html.html()))
            logging.info(f"List restaurants: found {len(tool_output)} restaurants. Task completed.")
        else:
            messages.append(AIMessage(content="<p>No restaurants found.</p>"))
            logging.info("List restaurants: No restaurants found. Task completed.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None

    elif current_task == "search_availability" and isinstance(tool_output, dict) and "results" in tool_output:
        availabilities = tool_output["results"]
        if availabilities:
            columns = [key.replace("_", " ").title() for key in availabilities[0].keys()] if availabilities else ["Availability Details"]
            availability_html = HTMLTableCreator(columns)
            for slot in availabilities:
                row_data = [slot.get(key, "N/A") for key in availabilities[0].keys()]
                availability_html.add_row(row_data)
            messages.append(AIMessage(content="<p>Here are the available slots:</p>" + availability_html.html()))
            logging.info(f"Search availability: found {len(availabilities)} slots. Task completed.")
        else:
            messages.append(AIMessage(content="<p>No availability found for the given criteria.</p>"))
            logging.info("Search availability: No availability found. Task completed.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None

    elif current_task == "create_booking" and isinstance(tool_output, dict) and "booking_reference" in tool_output:
        booking_ref = tool_output.get("booking_reference", "N/A")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking successfully created for <i>{restaurant_name}</i> with reference <b>{booking_ref}</b>.</p>"))
        logging.info(f"Booking created: {booking_ref}. Task completed.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None

    elif current_task == "cancel_booking" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        booking_ref = tool_parameters.get("booking_reference", "the booking")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking <b>{booking_ref}</b> for <i>{restaurant_name}</i> has been successfully cancelled.</p>"))
        logging.info(f"Booking cancelled: {booking_ref}. Task completed.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None

    elif current_task == "update_booking_details" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        booking_ref = tool_parameters.get("booking_reference", "the booking")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking <b>{booking_ref}</b> for <i>{restaurant_name}</i> has been successfully updated.</p>"))
        logging.info(f"Booking updated: {booking_ref}. Task completed.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None

    else:
        messages.append(AIMessage(content=f"<p>The {current_task} operation completed with result: {tool_output}</p>"))
        logging.info(f"Generic tool completion for {current_task}. State reset.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
    
    logging.debug(f"State after call_tool processing: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}")
    return state

# --- LangGraph Workflow Definition ---
# Configure SQLite checkpointer
# Explicitly get the SqliteSaver instance by entering the context manager
_memory_saver_context = SqliteSaver.from_conn_string(":memory:")
memory_saver = _memory_saver_context.__enter__() # Get the actual saver instance

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("call_tool", call_tool)

# Define entry point
workflow.add_edge(START, "agent")

# Define conditional transitions from the agent node
def route_agent_action(state: AgentState):
    logging.info(f"--- Entering route_agent_action ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    logging.debug(f"Route_agent_action: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}")
    logging.debug(f"Last message type: {type(last_message)}, has tool_calls: {getattr(last_message, 'tool_calls', 'N/A')}")

    # 1. If awaiting confirmation, stay in agent
    if state.get("sub_task_state") == "awaiting_confirmation":
        logging.info("Routing: Awaiting confirmation -> agent")
        return "agent"

    # 2. If the last message from the AI contains tool calls (i.e., agent_node suggested a tool)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_name_from_llm = last_message.tool_calls[0]['name']
        
        # Determine the task name for parameter check: use current_task if set, else the one from LLM's tool call
        # This handles both initial tool suggestion (current_task is None) and subsequent parameter collection
        task_for_param_check = state.get("current_task") or tool_name_from_llm
        
        # If the LLM's suggested tool aligns with the current task or is initiating a new one
        if state.get("current_task") == tool_name_from_llm or state.get("current_task") is None:
            missing = get_missing_parameters(task_for_param_check, state.get("tool_parameters", {}))
            
            if not missing and state.get("sub_task_state") is None:
                logging.info(f"Routing: Tool call suggested, all params collected, no sub-state -> call_tool ({tool_name_from_llm})")
                return "call_tool"
            else:
                logging.info(f"Routing: Tool call suggested, but missing params ({missing}) or in sub-state ({state.get('sub_task_state')}) -> agent")
                return "agent"
        else:
            # This is an unexpected scenario where LLM suggested a tool different from current_task.
            # Log and route back to agent for re-evaluation.
            logging.warning(f"Routing: LLM suggested tool '{tool_name_from_llm}' but current_task is '{state.get('current_task')}'. Routing back to agent.")
            return "agent"

    # 3. If the last message from the AI is a conversational response (no tool calls)
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # If there's an ongoing task that explicitly requires more conversational input (like selecting a booking,
        # or specifying update details, or cancellation reason)
        if state.get("current_task") is not None and state.get("sub_task_state") in ["awaiting_selection", "awaiting_details", "awaiting_cancellation_reason"]:
            logging.info(f"Routing: Conversational AI response, ongoing task {state.get('current_task')} in sub-state {state.get('sub_task_state')} -> agent")
            return "agent"
        else:
            # If no active task or specific multi-step sub-state needing continuation, the turn is complete.
            logging.info("Routing: Conversational AI response, no ongoing task or sub-state needing more input -> END")
            return END

    # 4. Fallback for other cases (e.g., initial HumanMessage, or if agent_node somehow didn't produce an AIMessage)
    # If the current task is None, we need the agent to figure out the intent.
    # If a HumanMessage just came in, and agent_node has not yet processed it into an AIMessage
    # (though in this graph, agent_node always adds an AIMessage before returning)
    logging.info("Routing: Fallback to agent (e.g., initial human input, or state not fitting above conditions).")
    return "agent"

workflow.add_conditional_edges(
    "agent",
    route_agent_action,
    {
        "call_tool": "call_tool",
        "agent": "agent", # Loop back to agent for more input or internal processing
        END: END, # Add END as a possible transition
    },
)

# Define conditional transitions from the call_tool node
def route_after_tool_call(state: AgentState):
    logging.info(f"--- Entering route_after_tool_call ---")
    logging.debug(f"State after tool call: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}")
    """
    Decides the next step after a tool call.
    If a sub_task_state is set, it means the agent needs to continue interaction.
    Otherwise, the task is complete for this turn.
    """
    # If a sub_task_state is active, return to agent for further processing (e.g., awaiting selection)
    if state.get("sub_task_state") is not None:
        logging.info("Routing: After tool call, sub_task_state is active -> agent")
        return "agent" 
    else:
        # Task is fully completed, end the current turn.
        logging.info("Routing: After tool call, task completed -> END")
        return END

workflow.add_conditional_edges(
    "call_tool",
    route_after_tool_call,
    {
        "agent": "agent", # Go back to agent for further interaction in a multi-step task
        END: END,         # End the graph execution for this turn if the task is complete
    },
)

# Compile the graph with the checkpointer
app = workflow.compile(checkpointer=memory_saver) 

# Define initial state for new conversations (outside the with block, as 'app' is now defined)
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

if __name__ == "__main__":
    all_ai_tools

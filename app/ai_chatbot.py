import functools, operator, re, inspect, os, dotenv, logging
from typing import Dict, List, Any, Optional, Union, Tuple, Literal, TypedDict
from datetime import date, time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools.structured import BaseTool, StructuredTool
from langgraph.graph import StateGraph, END, START
from langchain.chat_models.base import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

# Import your existing AI tools interface
from app.ai_tools import AIToolCallingInterface

# For intent classification
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = dotenv.get_key(".env","gemini_api") # Ensure this matches your .env key name

# Collect all actual Tool instances from AIToolCallingInterface
all_ai_tools_map = {tool.name: tool for tool in (getattr(AIToolCallingInterface, key) for key in dir(AIToolCallingInterface) if isinstance(getattr(AIToolCallingInterface, key), StructuredTool))}

# Define tool sets for each agent
# Main Agent handles everything *except* direct update/cancel calls; it redirects them to find_customer_bookings
main_agent_tool_names = ["list_restaurants_tool", "search_availability_tool", "create_booking_tool", "find_customer_bookings_tool"]
# Booking Management Agent handles the multi-step update/cancel process
booking_management_agent_tool_names = ["update_booking_details_tool", "cancel_booking_tool", "list_cancellation_reasons_tool"]

main_agent_tools = [all_ai_tools_map[name] for name in main_agent_tool_names]
booking_management_agent_tools = [all_ai_tools_map[name] for name in booking_management_agent_tool_names]

logging.debug("Main Agent Tools:")
for tool in main_agent_tools:
    logging.debug(f"- {tool.name}")
logging.debug("Booking Management Agent Tools:")
for tool in booking_management_agent_tools:
    logging.debug(f"- {tool.name}")

today = date.today()
promt_spec = ("You are a helpful and efficient restaurant booking assistant. Your primary goal is to assist users with queries, including:",
              f"""When responding, use the available tools to fulfill user requests. Here's how:

                1. Prioritize Tool Usage: If a user's request can be directly addressed by one of your tools, propose or call that tool.
                2. Conversational Fallback: If a request does not fit any tool, respond conversationally and offer alternative help.

                The day when user speaks is: {today}, {['Mon','Tues','Wednes','Thurs','Fri','Satur','Sun'][today.weekday()]}day. 
                Use the same year, month, or day if user mentions another day.
            """)
# Define the ChatPromptTemplate to provide system instructions
main_prompt = ChatPromptTemplate.from_messages(
    [("system", promt_spec[0] + """
            - Listing available restaurants.
            - Searching for booking availability.
            - Creating new bookings.
            - Finding, updating, or canceling customer's existing bookings.
            """ + promt_spec[1]),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

manage_prompt = ChatPromptTemplate.from_messages(
    [("system", promt_spec[0] + """
            - Listing available restaurants.
            - Searching for booking availability.
            - Creating new bookings.
            - Finding, updating, or canceling customer's existing bookings.
            """ + promt_spec[1]),

        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Initialize the model once
model = init_chat_model(
    model="gemini-2.0-flash-lite",
    model_provider="google_genai"
)

# Create two separate chains, each bound to its respective set of tools
main_chain = main_prompt | model.bind_tools(main_agent_tools)
booking_management_chain = manage_prompt | model.bind_tools(booking_management_agent_tools)


# Define the AgentState - using 'messages' for LangGraph's checkpointer
class AgentState(TypedDict):
    messages: List[BaseMessage] # This will store the chat history
    current_task: Optional[str] # e.g., 'list_restaurants_tool', 'search_availability_tool', 'create_booking_tool', 'find_customer_bookings_tool', 'update_booking_details_tool', 'cancel_booking_tool'
    sub_task_state: Optional[str] # For multi-step tasks: 'awaiting_selection', 'awaiting_details', 'awaiting_cancellation_reason', 'awaiting_confirmation'
    tool_parameters: Dict[str, Any] # Parameters collected for the current task
    customer_email: Optional[str] # Store customer email for repeated use
    bookings_found: Optional[List[Dict[str, Any]]] # Store list of bookings found for selection
    selected_booking_ref: Optional[str] # The reference of the booking chosen for update/cancel
    selected_restaurant_name: Optional[str] # The restaurant_name chosen for update/cancel
    booking_to_modify: Optional[Dict[str, Any]] # Stores the entire booking object chosen for update/cancel
    # New field to indicate if an update/cancel flow was intended from main agent
    original_task_after_find_bookings: Optional[str]

# --- Intent Classifier Setup ---
class IntentClassifier:
    # Using a general purpose sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    intents = {
        "create_booking": ["book a table", "make a reservation", "I want to reserve a spot", "new booking"],
        "find_booking": ["find my booking", "where is my reservation", "check my existing booking", "my current reservations"],
        "update_booking": ["change my booking", "update my reservation", "modify my booking details", "alter my reservation"],
        "cancel_booking": ["cancel my booking", "I want to cancel a reservation", "delete my booking", "remove my reservation"],
        "list_restaurants": ["list restaurants", "show me restaurants", "what restaurants are there", "restaurant list"],
        "search_availability": ["check availability", "is there a table available", "show me free slots", "restaurant availability"]
    }        
    # Create a flat list of all example sentences and their corresponding labels
    sentences = []
    labels = []
    for label, examples in intents.items():
        sentences.extend(examples)
        labels.extend([label] * len(examples))
    
    sentence_embeddings = model.encode(sentences)
    sentence_to_label_map = {s: l for s, l in zip(sentences, labels)}

    @classmethod
    def classify(cls, text: str, threshold: float = 0.5) -> Optional[str]:
        query_embedding = IntentClassifier.model.encode(text)
        cosine_scores = util.cos_sim(query_embedding, cls.sentence_embeddings)[0].cpu().numpy()
        
        # Find the max score and its corresponding original sentence and label
        max_score_idx = np.argmax(cosine_scores)
        max_score = cosine_scores[max_score_idx]
        
        predicted_sentence = list(cls.sentence_to_label_map.keys())[max_score_idx]
        predicted_label = cls.sentence_to_label_map[predicted_sentence]

        logging.info(f"Classifier result: Input='{text}', BestMatch='{predicted_sentence}' (Score={max_score:.2f}), PredictedLabel='{predicted_label}'")

        if max_score >= threshold:
            return predicted_label
        return None # No confident classification

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
    logging.debug(f"Checking missing parameters for: {tool_name} with collected: {collected_params}")
    
    # Determine which tool set to use based on the tool_name
    target_tool = all_ai_tools_map.get(tool_name)

    if not target_tool:
        logging.warning(f"Tool function not found for '{tool_name}' in any agent's tool set.")
        return []

    func = target_tool.func
    sig = inspect.signature(func)
    missing_params = []
    for param_name, param_info in sig.parameters.items():
        if param_name == 'self':
            continue
        
        # Check if the parameter is required (no default value) and not yet collected
        if param_info.default == inspect.Parameter.empty and param_name not in collected_params:
            # Specific exclusion for certain customer_ fields in create_booking_tool if not explicitly needed
            if tool_name == "create_booking_tool" and param_name.startswith("customer_") and param_name not in ["customer_email", "customer_first_name", "customer_surname", "customer_mobile"]:
                continue
            # customer_email for find_customer_bookings_tool is handled separately in agent_node when user is prompted for it.
            # Here, we only deem it missing if it's required by the tool but not in collected_params.
            if tool_name == "find_customer_bookings_tool" and param_name == "email": # The tool parameter is 'email', not 'customer_email'
                 if not collected_params.get("email"):
                    missing_params.append(param_name)
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

# Main Agent Node
def main_agent_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"--- Entering main_agent_node ---")
    logging.debug(f"Main AgentState on entry: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}, tool_parameters={state.get('tool_parameters')}")

    messages = state.get("messages", [])
    tool_parameters = state.get("tool_parameters", {})
    user_input = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""

    # Initial intent classification for the first user message or when starting a new task
    # This ensures the classifier drives update/cancel flows proactively.
    # Check if current_task is None AND there are no tool parameters or sub-task states set,
    # and it's a new human message (not a follow-up from an internal step)
    is_initial_query_or_reset = (state.get("current_task") is None and 
                                 state.get("sub_task_state") is None and 
                                 not state.get("tool_parameters") and
                                 isinstance(messages[-1], HumanMessage))

    if is_initial_query_or_reset:
        classified_intent = IntentClassifier.classify(user_input, threshold=0.6) # Increase threshold for more confidence
        logging.info(f"Initial classification result: {classified_intent}")

        if classified_intent in ["update_booking", "cancel_booking"]:
            target_tool_name = classified_intent + "_tool" # e.g., "update_booking_details_tool"
            state["current_task"] = "find_customer_bookings_tool"
            state["original_task_after_find_bookings"] = target_tool_name
            
            # If customer email is not known, prompt for it for find_customer_bookings_tool
            if not state.get("customer_email"):
                messages.append(AIMessage(content=f"<p>To {classified_intent.replace('_', ' ')} your booking, I need your email address. Could you please provide it?</p>"))
                logging.info(f"Classifier routed to find_customer_bookings for {target_tool_name}, but email missing. Prompting.")
                return state # Stay in main_agent to collect email
            else:
                # If email is known, pre-fill tool_parameters for find_customer_bookings_tool
                tool_parameters["email"] = state["customer_email"] # The parameter for find_customer_bookings_tool is 'email'
                state["tool_parameters"] = tool_parameters
                messages.append(AIMessage(content=f"<p>Understood. I'll search for your bookings using your email to help you {classified_intent.replace('_', ' ')}.</p>"))
                logging.info(f"Classifier routed to find_customer_bookings for {target_tool_name}, email known. Ready for tool call.")
                return state # The router will send to call_tool.

    # Handle email collection if currently in find_customer_bookings flow and email is missing
    # This path is taken after the initial classifier redirection if email was missing
    if state.get("current_task") == "find_customer_bookings_tool" and not state.get("tool_parameters", {}).get("email"):
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_input)
        if email_match:
            state["customer_email"] = email_match.group(0)
            tool_parameters["email"] = state["customer_email"] # Parameter for find_customer_bookings_tool is 'email'
            state["tool_parameters"] = tool_parameters
            messages.append(AIMessage(content=f"<p>Thank you. I have your email as {state['customer_email']}.</p>"))
            logging.info("Collected email for find_customer_bookings.")
            return state # The router will send to call_tool as email is now present
        else:
            # If we were expecting email but didn't get it, reprompt.
            if len(messages) > 1 and "email" in messages[-2].content.lower(): # Crude check if we just asked for email
                messages.append(AIMessage(content="<p>That doesn't look like a valid email. Please provide your email address to find your bookings.</p>"))
            else: # If it's a new turn but we're still waiting for email
                messages.append(AIMessage(content="<p>Please provide your email address so I can find your bookings.</p>"))
            return state # Stay in main agent to collect email


    # If not explicitly redirected by the classifier for update/cancel, or if email was just collected for find_customer_bookings,
    # proceed with LLM invocation for general intent or to extract tool parameters.
    context_messages = list(messages) 
    # Add a synthetic message to provide LLM with customer email if known and not already in tool_parameters
    if state.get("customer_email") and "email" not in tool_parameters:
         context_messages.insert(0, AIMessage(content=f"User's known email: {state['customer_email']}."))
    
    response = main_chain.invoke({"messages": context_messages})
    messages.append(response) # Append LLM's full response (including tool calls or conversational content)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call['name']
        params = tool_call['args']
        logging.info(f"Main Agent LLM suggested tool call: {tool_name} with params: {params}")

        # If LLM *also* suggested email, ensure it's captured (e.g., for create_booking_tool)
        if params.get("customer_email") and not state.get("customer_email"):
            state["customer_email"] = params["customer_email"]
            logging.info(f"Customer email '{state['customer_email']}' extracted by LLM and persisted.")
        
        tool_parameters.update(params)
        state["tool_parameters"] = tool_parameters
        state["current_task"] = tool_name

        # If LLM *directly* suggested update/cancel (not via classifier, or classifier wasn't confident)
        if tool_name in ["update_booking_details_tool", "cancel_booking_tool"]:
            logging.info(f"Main Agent LLM detected '{tool_name}' intent directly. Redirecting to find_customer_bookings_tool.")
            state["original_task_after_find_bookings"] = tool_name
            state["current_task"] = "find_customer_bookings_tool"
            # Ensure email is passed for find_customer_bookings_tool
            if not state.get("tool_parameters", {}).get("email") and state.get("customer_email"):
                state["tool_parameters"]["email"] = state["customer_email"]
            elif not state.get("tool_parameters", {}).get("email"):
                 messages.append(AIMessage(content="<p>To manage your bookings, I need your email address. Could you please provide it?</p>"))
                 return state # Stay in main_agent to collect email
            return state
        
        # For other tools (list, search, create)
        missing = get_missing_parameters(tool_name, tool_parameters)
        if missing:
            columns = ["Parameter"]
            missing_html = HTMLTableCreator(columns)
            for m in missing:
                missing_html.add_row([m])
            messages.append(AIMessage(content=f"<p>I need more information for your <i>{tool_name}</i> request. Please provide:</p>{missing_html.html()}"))
            logging.info(f"Still missing parameters for {tool_name}: {missing}. Staying in main_agent.")
        else:
            logging.info(f"All parameters collected for {tool_name}. Ready for tool call or confirmation.")
            if tool_name == "create_booking_tool":
                state["sub_task_state"] = "awaiting_confirmation"
                messages.append(AIMessage(content=f"<p>Confirm you wish to create a booking for <i>{tool_parameters.get('restaurant_name', 'the restaurant')}</i> on {tool_parameters.get('visit_date')} at {tool_parameters.get('visit_time')} for {tool_parameters.get('party_size')} people? Please confirm.</p>"))
        return state
    else: # LLM did not suggest a tool call (conversational response)
        logging.info("Main Agent: Model did not suggest a tool call. Conversational response.")
        return state

# Booking Management Agent Node
def booking_management_agent_node(state: AgentState) -> Dict[str, Any]:
    logging.info(f"--- Entering booking_management_agent_node ---")
    logging.debug(f"Booking Management AgentState on entry: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}, tool_parameters={state.get('tool_parameters')}, bookings_found={bool(state.get('bookings_found'))}")

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
        logging.info(f"Booking Management Agent in 'awaiting_confirmation'. User input: '{user_input}'")
        if is_confirmation(user_input):
            messages.append(AIMessage(content="<p>Acknowledged. Proceeding with the action...</p>"))
            state["sub_task_state"] = None # Clear sub_task_state so call_tool can be triggered
            logging.info("Confirmation received. Setting sub_task_state to None.")
            return state # Route to call_tool
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
            state["original_task_after_find_bookings"] = None
            state["customer_email"] = None # Reset customer email on full cancellation
            return state # Route to END
        else:
            messages.append(AIMessage(content="<p>Please confirm with 'yes' or 'no'.</p>"))
            logging.info("Invalid confirmation input. Asking user to confirm again.")
            return state # Stay in booking_management_agent

    # --- Logic for handling booking selection (for update/cancel) ---
    if sub_task_state == "awaiting_selection":
        logging.info(f"Booking Management Agent in 'awaiting_selection' state for {current_task}. User input: '{user_input}'")
        if bookings_found:
            num_match = re.search(r"\b\d+", user_input)
            if num_match:
                try:
                    selected_index = int(num_match.group()) - 1
                    if 0 <= selected_index < len(bookings_found):
                        selected_booking = bookings_found[selected_index]
                        state["selected_booking_ref"] = selected_booking["booking_reference"]
                        state["selected_restaurant_name"] = selected_booking["restaurant_name"]
                        # Set tool_parameters for the actual update/cancel call
                        tool_parameters["restaurant_name"] = selected_booking["restaurant_name"]
                        tool_parameters["booking_reference"] = selected_booking["booking_reference"]
                        state["booking_to_modify"] = selected_booking
                        state["tool_parameters"] = tool_parameters # Update state's tool_parameters

                        if current_task == "cancel_booking_tool":
                            state["sub_task_state"] = "awaiting_cancellation_reason"
                            # Call list_cancellation_reasons_tool directly here
                            cancellation_reasons = AIToolCallingInterface.list_cancellation_reasons_tool()
                            columns = ["Reason ID", "Title", "Description"]
                            reason_html = HTMLTableCreator(columns)
                            for r in cancellation_reasons:
                                reason_html.add_row([r.get("id", "N/A"), r.get("title", "N/A"), r.get("description", "N/A")])
                            
                            messages.append(AIMessage(content=f"<p>You've selected booking <b>{selected_booking['booking_reference']}</b> for <i>{selected_booking['restaurant_name']}</i>. Please provide the ID of the cancellation reason from the following table: </p>"+reason_html.html()))
                            logging.info(f"Selected booking {selected_booking['booking_reference']} for cancellation. Transitioning to 'awaiting_cancellation_reason'.")
                        elif current_task == "update_booking_details_tool":
                            state["sub_task_state"] = "awaiting_details"
                            messages.append(AIMessage(content=f"<p>You've selected booking <b>{selected_booking['booking_reference']}</b> for <i>{selected_booking['restaurant_name']}</i>. What details would you like to update (e.g., visit_date, visit_time, party_size, special_requests)?</p>"))
                            logging.info(f"Selected booking {selected_booking['booking_reference']} for update. Transitioning to 'awaiting_details'.")
                        return state # Stay in booking_management_agent
                    else:
                        messages.append(AIMessage(content=f"<p>That booking number is out of range. Please select a number between 1 and {len(bookings_found)}.</p>"))
                        logging.info("Invalid booking number selected (out of range).")
                        return state # Stay in booking_management_agent
                except ValueError:
                    messages.append(AIMessage(content="<p>I couldn't identify the booking you wish to modify. Please specify the booking number as an integer value (1, 2, 3...).</p>"))
                    logging.info("Invalid booking number input (not an integer).")
                    return state # Stay in booking_management_agent
            else:
                messages.append(AIMessage(content="<p>I couldn't identify the booking you wish to modify. Please specify the booking number as an integer value (1, 2, 3...).</p>"))
                logging.info("No booking number identified in user input.")
                return state # Stay in booking_management_agent

    # --- Logic for collecting cancellation reason ID ---
    if sub_task_state == "awaiting_cancellation_reason":
        logging.info(f"Booking Management Agent in 'awaiting_cancellation_reason'. User input: '{user_input}'")
        try:
            reason_id_match = re.search(r'\b\d+', user_input)
            if reason_id_match:
                reason_id = int(reason_id_match.group())
                tool_parameters["cancellation_reason_id"] = reason_id
                state["tool_parameters"] = tool_parameters # Update state's tool_parameters
                state["sub_task_state"] = "awaiting_confirmation"
                messages.append(AIMessage(content=f"<p>Are you sure you want to cancel booking <b>{selected_booking_ref}</b> for <i>{selected_restaurant_name}</i> with reason No.<b>{reason_id}</b>? Please confirm.</p>"))
                logging.info(f"Cancellation reason ID '{reason_id}' collected. Transitioning to 'awaiting_confirmation'.")
                return state # Stay in booking_management_agent
            else:
                messages.append(AIMessage(content="<p>Please provide a valid cancellation reason ID (a number).</p>"))
                logging.info("No valid cancellation reason ID found in input.")
                return state # Stay in booking_management_agent
        except (AttributeError, ValueError):
            messages.append(AIMessage(content="<p>Please provide a valid cancellation reason ID (a number).</p>"))
            logging.info("Error parsing cancellation reason ID.")
            return state # Stay in booking_management_agent

    # --- Logic for collecting update details ---
    if sub_task_state == "awaiting_details":
        logging.info(f"Booking Management Agent in 'awaiting_details' for update_booking_details_tool. User input: '{user_input}')")
        response = booking_management_chain.invoke({"messages": messages}) # Use booking_management_chain here
        messages.append(response) # Append LLM's response

        if response.tool_calls and response.tool_calls[0]['name'] == current_task:
            new_params = response.tool_calls[0]['args']
            
            # Remove parameters that are part of the booking identity, not for update
            for old_param in ["restaurant_name", "booking_reference"]:
                if old_param in new_params:
                    new_params.pop(old_param)

            if new_params:
                tool_parameters.update(new_params)
                state["tool_parameters"] = tool_parameters # Update state's tool_parameters
                
                messages.append(AIMessage(content=f"<p>Confirm you wish to update booking <b>{selected_booking_ref}</b> for <i>{selected_restaurant_name}</i> with the following changes: </p>{HTMLTableCreator().from_dict(new_params)}"))
                state["sub_task_state"] = "awaiting_confirmation"
                logging.info(f"Update parameters extracted: {new_params}. Transitioning to 'awaiting_confirmation'.")
                return state # Stay in booking_management_agent
            else:
                messages.append(AIMessage(content="<p>I couldn't detect any specific updates. What exactly would you like to update? Please specify (e.g., 'change date to 2025-01-01', 'party size to 5').</p>"))
                logging.info("Model did not extract new update parameters.")
                return state # Stay in booking_management_agent
        else:
            messages.append(AIMessage(content="<p>What exactly would you like to update? Please specify (e.g., 'change date to 2025-01-01', 'party size to 5').</p>"))
            logging.info("Model did not suggest tool call for update details or suggested wrong tool.")
            return state # Stay in booking_management_agent

    # Fallback/initial entry point for booking management agent
    logging.warning("Booking Management Agent: Reached fallback/initial state. This should ideally mean we're awaiting user input related to selected booking.")
    messages.append(AIMessage(content="<p>I'm ready to help you manage your booking. Please tell me what you'd like to do (e.g., 'update visit time', 'cancel reservation').</p>"))
    return state


# Node for executing the tool call
def call_tool(state: AgentState) -> Dict[str, Any]:
    logging.info(f"--- Entering call_tool node ---")
    logging.debug(f"State on entry: current_task={state.get('current_task')}, tool_parameters={state.get('tool_parameters')}")
    current_task = state["current_task"]
    tool_parameters = state["tool_parameters"]
    messages = state["messages"]

    selected_tool = all_ai_tools_map.get(current_task)
    selected_tool_func = selected_tool.func if selected_tool else None

    tool_output = None
    if not selected_tool_func:
        tool_output = f"Error: Tool '{current_task}' not found in AIToolCallingInterface."
        logging.error(f"Tool function not found: {current_task}")
    else:
        try:
            invoke_params = tool_parameters.copy() 
            # Type conversion for date/time objects for tool invocation
            for param, value in invoke_params.items():
                if param == "customer_email": # LLM might extract this. Tool expects 'email'
                    invoke_params["email"] = invoke_params.pop("customer_email")

            logging.info(f"Calling tool '{current_task}' with parameters: {invoke_params}")
            tool_output = selected_tool_func(**invoke_params)
            
            if isinstance(tool_output, dict) and "error" in tool_output:
                tool_output = f"Error: {tool_output['error']}"
                logging.error(f"Tool '{current_task}' returned an error: {tool_output}")
            elif isinstance(tool_output, list) and not tool_output and current_task != "list_cancellation_reasons_tool":
                tool_output = "No results found."
                logging.info(f"Tool '{current_task}' returned no results.")

        except Exception as e:
            tool_output = f"Error executing tool '{current_task}': {str(e)}"
            logging.exception(f"Exception during tool execution for '{current_task}'")

    tool_call_id = None
    # Find the latest AIMessage with tool_calls and match the current tool
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc['name'] == current_task:
                    tool_call_id = tc['id']
                    break
            if tool_call_id:
                break
    
    # Append the ToolMessage containing the output of the tool execution
    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id))
    logging.debug(f"ToolMessage added: {str(tool_output)[:100]}...")

    # --- Post-tool-call state management and response generation ---
    if current_task == "find_customer_bookings_tool" and not isinstance(tool_output, str) and "error" not in str(tool_output) and tool_output != "No results found.":
        bookings = tool_output
        state["bookings_found"] = bookings # Store found bookings
        
        original_task_after_find = state.get("original_task_after_find_bookings") # Retrieve original intent
        state["original_task_after_find_bookings"] = None # Clear it after use

        if bookings:
            columns = ["#"] + [key.replace("_", " ").title() for key in bookings[0].keys()] if bookings else ["#", "Booking Details"]
            bookings_html = HTMLTableCreator(columns)
            for i, booking in enumerate(bookings):
                row_data = [i+1] + [booking.get(key, "N/A") for key in bookings[0].keys()]
                bookings_html.add_row(row_data)
            
            if original_task_after_find in ["update_booking_details_tool", "cancel_booking_tool"]:
                state["current_task"] = original_task_after_find # Set the current_task to the actual update/cancel
                state["sub_task_state"] = "awaiting_selection"
                messages.append(AIMessage(content="<p>I found the following bookings for you:</p>" + bookings_html.html() + "<p>Please tell me which booking you'd like to update/cancel by its number (e.g., '1' for the first one).</p>"))
                logging.info(f"Find customer bookings: found {len(bookings)} bookings for {original_task_after_find}. Transitioning to 'awaiting_selection' in booking_management_agent.")
                # The router will send to booking_management_agent
            else:
                messages.append(AIMessage(content="<p>Here are your bookings:</p>" + bookings_html.html()))
                # Task completed, reset state for next turn
                state["current_task"] = None
                state["tool_parameters"] = {}
                state["sub_task_state"] = None
                state["bookings_found"] = None
                state["selected_booking_ref"] = None
                state["selected_restaurant_name"] = None
                state["booking_to_modify"] = None
                state["original_task_after_find_bookings"] = None
                logging.info("Find customer bookings: found bookings. Task completed, state reset for main agent.")

        else:
            messages.append(AIMessage(content="<p>No bookings found for the provided criteria.</p>"))
            # Task completed with no results, reset state
            state["current_task"] = None
            state["tool_parameters"] = {}
            state["sub_task_state"] = None
            state["bookings_found"] = None
            state["selected_booking_ref"] = None
            state["selected_restaurant_name"] = None
            state["booking_to_modify"] = None
            state["original_task_after_find_bookings"] = None
            logging.info("Find customer bookings: No bookings found. Task completed, state reset.")

    elif current_task == "list_restaurants_tool" and isinstance(tool_output, list) and not isinstance(tool_output, str):
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
        # Task completed, reset state
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None


    elif current_task == "search_availability_tool" and isinstance(tool_output, dict) and "results" in tool_output:
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
        # Task completed, reset state
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None


    elif current_task == "create_booking_tool" and isinstance(tool_output, dict) and "booking_reference" in tool_output:
        booking_ref = tool_output.get("booking_reference", "N/A")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking successfully created for <i>{restaurant_name}</i> with reference <b>{booking_ref}</b>.</p>"))
        logging.info(f"Booking created: {booking_ref}. Task completed.")
        # Task completed, reset state
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None


    elif current_task == "cancel_booking_tool" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        booking_ref = tool_parameters.get("booking_reference", "the booking")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking <b>{booking_ref}</b> for <i>{restaurant_name}</i> has been successfully cancelled.</p>"))
        logging.info(f"Booking cancelled: {booking_ref}. Task completed.")
        # Task completed, reset state
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None
        state["customer_email"] = None # Reset customer email after a successful cancellation

    elif current_task == "update_booking_details_tool" and isinstance(tool_output, dict) and tool_output.get("status") == "success":
        booking_ref = tool_parameters.get("booking_reference", "the booking")
        restaurant_name = tool_parameters.get("restaurant_name", "the restaurant")
        messages.append(AIMessage(content=f"<p>Booking <b>{booking_ref}</b> for <i>{restaurant_name}</i> has been successfully updated.</p>"))
        logging.info(f"Booking updated: {booking_ref}. Task completed.")
        # Task completed, reset state
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None

    elif isinstance(tool_output, str) and "Error:" in tool_output:
        messages.append(AIMessage(content=f"<p>There was an error with your request: {tool_output}</p>"))
        # Error occurred, reset state for new interaction
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None
        state["customer_email"] = None # Reset customer email on error to allow re-entry
        logging.error(f"Error handling for {current_task}: {tool_output}")

    else:
        # Generic tool output handling or when tool output is not a recognized success/failure pattern
        messages.append(AIMessage(content=f"<p>The {current_task} operation completed with result: {tool_output}</p>"))
        logging.info(f"Generic tool completion for {current_task}. State reset.")
        state["current_task"] = None
        state["tool_parameters"] = {}
        state["sub_task_state"] = None
        state["bookings_found"] = None
        state["selected_booking_ref"] = None
        state["selected_restaurant_name"] = None
        state["booking_to_modify"] = None
        state["original_task_after_find_bookings"] = None
    
    logging.debug(f"State after call_tool processing: current_task={state.get('current_task')}, sub_task_state={state.get('sub_task_state')}")
    return state

# --- LangGraph Workflow Definition ---
# Configure SQLite checkpointer
_memory_saver_context = SqliteSaver.from_conn_string(":memory:")
memory_saver = _memory_saver_context.__enter__() # Get the actual saver instance

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("main_agent", main_agent_node)
workflow.add_node("booking_management_agent", booking_management_agent_node)
workflow.add_node("call_tool", call_tool)

# Define entry point
workflow.add_edge(START, "main_agent")

# Define conditional transitions from main_agent node
def route_main_agent_action(state: AgentState):
    logging.info(f"--- Entering route_main_agent_action ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    current_task = state.get("current_task")
    sub_task_state = state.get("sub_task_state")
    tool_parameters = state.get("tool_parameters", {})
    logging.debug(f"Route_main_agent_action: current_task={current_task}, sub_task_state={sub_task_state}, last_msg_type={type(last_message)}, tool_calls={getattr(last_message, 'tool_calls', 'N/A')}")

    # If main agent just prompted for email for find_customer_bookings, stay in main_agent
    if current_task == "find_customer_bookings_tool" and not tool_parameters.get("email"):
        logging.info("Routing from main_agent: Awaiting customer email for find_customer_bookings. Staying in main_agent.")
        return "main_agent"

    # If main agent has set current_task to find_customer_bookings_tool and email is now available, go to call_tool
    if current_task == "find_customer_bookings_tool" and tool_parameters.get("email"):
        logging.info("Routing from main_agent: Ready to call find_customer_bookings_tool. Routing to call_tool.")
        return "call_tool"

    # If the last message from the main agent contains tool calls (for non-update/cancel tasks)
    if isinstance(last_message, AIMessage) and last_message.tool_calls and last_message.tool_calls[0]['name'] in main_agent_tool_names:
        tool_name = last_message.tool_calls[0]['name']
        missing = get_missing_parameters(tool_name, tool_parameters)
        
        if not missing and sub_task_state != "awaiting_confirmation":
            logging.info(f"Routing from main_agent: All params collected for {tool_name} -> call_tool")
            return "call_tool"
        elif sub_task_state == "awaiting_confirmation":
            logging.info(f"Routing from main_agent: Awaiting confirmation for {tool_name} -> main_agent (stay)")
            return "main_agent"
        else: # Missing parameters
            logging.info(f"Routing from main_agent: Missing params for {tool_name} -> main_agent (stay)")
            return "main_agent"

    # If main agent responded conversationally and task is not fully handled
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # If no active task or sub_task_state, then the turn is complete.
        if current_task is None and sub_task_state is None:
            logging.info("Routing from main_agent: Conversational response, no active task -> END")
            return END
        else: # Conversational response in an ongoing task, may need more input
            logging.info(f"Routing from main_agent: Conversational response, active task {current_task} -> main_agent (stay)")
            return "main_agent"

    logging.warning("Routing from main_agent: Fallback to main_agent.")
    return "main_agent"

workflow.add_conditional_edges(
    "main_agent",
    route_main_agent_action,
    {
        "call_tool": "call_tool",
        "main_agent": "main_agent",
        END: END,
    },
)

# Define conditional transitions from booking_management_agent node
def route_booking_management_agent_action(state: AgentState):
    logging.info(f"--- Entering route_booking_management_agent_action ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    current_task = state.get("current_task")
    sub_task_state = state.get("sub_task_state")
    tool_parameters = state.get("tool_parameters", {})
    logging.debug(f"Route_booking_management_agent_action: current_task={current_task}, sub_task_state={sub_task_state}, last_msg_type={type(last_message)}, tool_calls={getattr(last_message, 'tool_calls', 'N/A')}")

    # If awaiting confirmation, stay in booking_management_agent
    if sub_task_state == "awaiting_confirmation":
        logging.info("Routing from booking_management_agent: Awaiting confirmation -> booking_management_agent")
        return "booking_management_agent"

    # If the last message from the AI contains tool calls (e.g., for update details)
    if isinstance(last_message, AIMessage) and last_message.tool_calls and last_message.tool_calls[0]['name'] in booking_management_agent_tool_names:
        tool_name = last_message.tool_calls[0]['name']
        missing = get_missing_parameters(tool_name, tool_parameters)
        
        if not missing and sub_task_state != "awaiting_confirmation":
            logging.info(f"Routing from booking_management_agent: All params collected for {tool_name} -> call_tool")
            return "call_tool"
        else: # Missing parameters or awaiting confirmation (which is handled above)
            logging.info(f"Routing from booking_management_agent: Missing params for {tool_name} -> booking_management_agent (stay)")
            return "booking_management_agent"

    # If the last message is a conversational response
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # If in a sub-state requiring user input (selection, details, cancellation reason)
        if sub_task_state in ["awaiting_selection", "awaiting_details", "awaiting_cancellation_reason"]:
            logging.info(f"Routing from booking_management_agent: Conversational response, in sub-state {sub_task_state} -> booking_management_agent (stay)")
            return "booking_management_agent"
        else:
            # If the task is conceptually complete (e.g., user said "no" to confirmation and state was reset)
            logging.info("Routing from booking_management_agent: Conversational response, task completed or user cancelled -> END")
            return END

    logging.warning("Routing from booking_management_agent: Fallback to booking_management_agent.")
    return "booking_management_agent"


workflow.add_conditional_edges(
    "booking_management_agent",
    route_booking_management_agent_action,
    {
        "call_tool": "call_tool",
        "booking_management_agent": "booking_management_agent",
        END: END,
    },
)

# Define conditional transitions from call_tool node
def route_after_tool_call(state: AgentState):
    logging.info(f"--- Entering route_after_tool_call ---")
    current_task = state.get("current_task")
    sub_task_state = state.get("sub_task_state")
    original_task_after_find = state.get("original_task_after_find_bookings")
    logging.debug(f"Route_after_tool_call: current_task={current_task}, sub_task_state={sub_task_state}, original_task_after_find_bookings={original_task_after_find}")

    if current_task == "find_customer_bookings_tool" and original_task_after_find in ["update_booking_details_tool", "cancel_booking_tool"]:
        # If find_customer_bookings was called as a precursor to update/cancel, route to booking_management_agent
        logging.info("Routing after call_tool: find_customer_bookings completed for update/cancel. Routing to booking_management_agent.")
        return "booking_management_agent"
    
    # If a sub_task_state is active (e.g., awaiting confirmation after parameter collection in booking_management_agent)
    if sub_task_state is not None:
        logging.info("Routing after call_tool: sub_task_state is active -> booking_management_agent")
        return "booking_management_agent" 
    
    # If the task completed (no sub_task_state or specific follow-up)
    logging.info("Routing after call_tool: Task completed -> END")
    return END

workflow.add_conditional_edges(
    "call_tool",
    route_after_tool_call,
    {
        "booking_management_agent": "booking_management_agent",
        END: END,         # End the graph execution for this turn if the task is complete
    },
)

# Compile the graph with the checkpointer
app = workflow.compile(checkpointer=memory_saver) 

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
    "booking_to_modify": None,
    "original_task_after_find_bookings": None
}

if __name__ == "__main__":
    print(SqliteSaver.__dict__.keys())
    raise KeyboardInterrupt()
    # Example usage:
    # Example 1: User wants to cancel a booking (email known, direct classification)
    thread_id_ex1 = "test-thread-ex1"
    state_ex1 = initial_agent_state.copy()
    state_ex1["customer_email"] = "testuser@example.com" # Pre-fill email for this example

    print("\n--- Example 1: User wants to cancel a booking (email known, direct classification) ---")
    inputs = {"messages": [HumanMessage(content="I need to cancel my reservation.")]}
    for s in app.stream(inputs, {"configurable": {"thread_id": thread_id_ex1, "thread_state": state_ex1}}):
        if "__end__" not in s:
            print(s)
    
    # Follow-up for Example 1: Select booking (simulate find_customer_bookings_tool output)
    current_state_ex1 = app.get_state({"configurable": {"thread_id": thread_id_ex1}})._values
    if not current_state_ex1.get("bookings_found"):
         current_state_ex1["bookings_found"] = [
             {"booking_reference": "REF123", "restaurant_name": "Fancy Feast", "visit_date": "2025-08-20", "visit_time": "19:00:00", "party_size": 2},
             {"booking_reference": "REF456", "restaurant_name": "Quick Bites", "visit_date": "2025-09-10", "visit_time": "12:30:00", "party_size": 4}
         ]
         # Ensure current_task is correctly set for booking_management_agent (it should be if routed from call_tool)
         current_state_ex1["current_task"] = "cancel_booking_tool"
         current_state_ex1["sub_task_state"] = "awaiting_selection"
         
    inputs_select_booking_ex1 = {"messages": [HumanMessage(content="I want to cancel booking number 1.")]}
    print("\n--- Example 1 follow-up: User selects booking 1 for cancellation ---")
    for s in app.stream(inputs_select_booking_ex1, {"configurable": {"thread_id": thread_id_ex1, "thread_state": current_state_ex1}}):
        if "__end__" not in s:
            print(s)
    
    # Follow-up for Example 1: Provide cancellation reason
    current_state_after_selection_ex1 = app.get_state({"configurable": {"thread_id": thread_id_ex1}})._values
    inputs_reason_ex1 = {"messages": [HumanMessage(content="Reason ID 3.")]}
    print("\n--- Example 1 follow-up: User provides cancellation reason (ID 3) ---")
    for s in app.stream(inputs_reason_ex1, {"configurable": {"thread_id": thread_id_ex1, "thread_state": current_state_after_selection_ex1}}):
        if "__end__" not in s:
            print(s)
    
    # Follow-up for Example 1: Confirm cancellation
    current_state_after_reason_ex1 = app.get_state({"configurable": {"thread_id": thread_id_ex1}})._values
    inputs_confirm_cancel_ex1 = {"messages": [HumanMessage(content="yes")]}
    print("\n--- Example 1 follow-up: User confirms cancellation ---")
    for s in app.stream(inputs_confirm_cancel_ex1, {"configurable": {"thread_id": thread_id_ex1, "thread_state": current_state_after_reason_ex1}}):
        if "__end__" not in s:
            print(s)

    # Example 2: User wants to create a booking (Main Agent task) - Classifier should not interfere much here
    print("\n--- Example 2: User wants to create a booking ---")
    thread_id_create = "test-thread-create-1"
    inputs_create = {"messages": [HumanMessage(content="Book a table for 4 at The Grand Restaurant on 2025-12-25 at 19:30, my email is newuser@example.com, first name New, surname User, mobile 123456789.")]}
    for s in app.stream(inputs_create, {"configurable": {"thread_id": thread_id_create, "thread_state": initial_agent_state}}):
        if "__end__" not in s:
            print(s)

    # Follow-up for Example 2: Confirm creation
    current_state_after_create_prompt = app.get_state({"configurable": {"thread_id": thread_id_create}})._values
    inputs_confirm_create = {"messages": [HumanMessage(content="yes")]}
    print("\n--- Example 2 follow-up: User confirms booking creation ---")
    for s in app.stream(inputs_confirm_create, {"configurable": {"thread_id": thread_id_create, "thread_state": current_state_after_create_prompt}}):
        if "__end__" not in s:
            print(s)

    # Example 3: User wants to update a booking (email unknown initially, then provide email)
    print("\n--- Example 3: User wants to update a booking (email unknown initially) ---")
    thread_id_update_flow = "test-thread-update-flow-1"
    inputs_update_initial = {"messages": [HumanMessage(content="I need to change my reservation.")]}
    for s in app.stream(inputs_update_initial, {"configurable": {"thread_id": thread_id_update_flow, "thread_state": initial_agent_state}}):
        if "__end__" not in s:
            print(s)
    
    # Provide email
    inputs_update_email = {"messages": [HumanMessage(content="My email is updateuser@example.com")]}
    print("\n--- Example 3 follow-up: Providing email ---")
    for s in app.stream(inputs_update_email, {"configurable": {"thread_id": thread_id_update_flow}}):
        if "__end__" not in s:
            print(s)

    # Simulate find_customer_bookings_tool output
    current_state_after_email = app.get_state({"configurable": {"thread_id": thread_id_update_flow}})._values
    if not current_state_after_email.get("bookings_found"):
        current_state_after_email["bookings_found"] = [
            {"booking_reference": "UPD801", "restaurant_name": "The Modern Bistro", "visit_date": "2025-10-01", "visit_time": "18:00:00", "party_size": 3},
            {"booking_reference": "UPD802", "restaurant_name": "The Modern Bistro", "visit_date": "2025-11-15", "visit_time": "20:00:00", "party_size": 2}
        ]
        current_state_after_email["current_task"] = "update_booking_details_tool"
        current_state_after_email["sub_task_state"] = "awaiting_selection"

    # Select booking
    inputs_update_select = {"messages": [HumanMessage(content="I want to change booking number 2.")]}
    print("\n--- Example 3 follow-up: User selects booking 2 for update ---")
    for s in app.stream(inputs_update_select, {"configurable": {"thread_id": thread_id_update_flow, "thread_state": current_state_after_email}}):
        if "__end__" not in s:
            print(s)
    
    # Provide update details
    current_state_after_select = app.get_state({"configurable": {"thread_id": thread_id_update_flow}})._values
    inputs_update_details = {"messages": [HumanMessage(content="Change party size to 4 and visit time to 19:00.")]}
    print("\n--- Example 3 follow-up: User provides update details ---")
    for s in app.stream(inputs_update_details, {"configurable": {"thread_id": thread_id_update_flow, "thread_state": current_state_after_select}}):
        if "__end__" not in s:
            print(s)

    # Confirm update
    current_state_after_details = app.get_state({"configurable": {"thread_id": thread_id_update_flow}})._values
    inputs_confirm_update = {"messages": [HumanMessage(content="yes")]}
    print("\n--- Example 3 follow-up: User confirms update ---")
    for s in app.stream(inputs_confirm_update, {"configurable": {"thread_id": thread_id_update_flow, "thread_state": current_state_after_details}}):
        if "__end__" not in s:
            print(s)
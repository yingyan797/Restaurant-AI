# app/ai_chatbot.py
import os, logging, dotenv, json, re
import copy # Import copy for initial_agent_state
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import date, time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # NEW: Import the checkpointer
from sentence_transformers import SentenceTransformer, util
from langchain.chat_models.base import init_chat_model
from app.ai_tools import AIToolCallingInterface as tools
import inspect
import numpy as np # Ensure numpy is imported for classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = dotenv.get_key(".env","gemini_api")

class AIModel:
    classifier = SentenceTransformer('all-MiniLM-L6-v2')
    llm = init_chat_model(
        model="gemini-2.0-flash-lite",
        model_provider="google_genai"
    )
    is_simulation = False
    print("This is simulation mode" if is_simulation else "Will call LLM")

    # Define intents and their associated keywords/phrases
    intent_phrases = {
        "list_restaurants_tool": ["list restaurants", "show me restaurants", "what restaurants do you have", "what are your restaurants"],
        "list_cancellation_reasons_tool": ["cancellation reasons", "why can I cancel", "list reasons to cancel", "cancellation policies"],
        "find_customer_bookings_tool": ["find my bookings", "show my reservations for a restaurant", "what bookings do I have", "check my bookings on a date", "find my bookings for several people"],
        "search_availability_tool": ["check restaurant availability", "is there a table available", "find an available time for the restaurant", "are there any tables available for the restaurant"],
        "create_booking_tool": ["make a booking", "create a reservation for the restaurant", "book a table on a date", "I want to reserve"],
        "cancel_booking_tool": ["cancel my booking", "cancel a reservation for the restaurant", "I want to cancel"],
        "update_booking_details_tool": ["change my booking", "update reservation to a new date", "modify my booking to restaurant", "I need to change my reservation detail"],
        "chitchat": ["hello", "hi", "how are you", "what can you do", "who are you", "tell me about yourself", "thanks", "thank you", "bye", "goodbye"]
    }

    # Pre-compute embeddings for intent phrases
    sentences = []
    labels = []
    for label, examples in intent_phrases.items():
        sentences.extend(examples)
        labels.extend([label] * len(examples))
    
    sentence_embeddings = classifier.encode(sentences)
    sentence_to_label_map = {s: l for s, l in zip(sentences, labels)}

    @classmethod
    def classify_intent(cls, user_query: str) -> str:
        query_embedding = cls.classifier.encode(user_query)
        cosine_scores = util.cos_sim(query_embedding, cls.sentence_embeddings)[0].cpu().numpy()
        
        max_score_idx = np.argmax(cosine_scores)
        max_score = cosine_scores[max_score_idx]
        normalized_scores = (cosine_scores - np.mean(cosine_scores)) / np.std(cosine_scores)
        max_norm_score = normalized_scores[max_score_idx]
        
        predicted_sentence = list(cls.sentence_to_label_map.keys())[max_score_idx]
        predicted_label = cls.sentence_to_label_map[predicted_sentence]

        logging.info(f"Classifier result: BestMatch='{predicted_sentence}' (Score={max_score:.2f} + {max_norm_score:.2f} std), PredictedLabel='{predicted_label}'")

        if max_score >= 0.6 or (max_score >= 0.4 and max_norm_score >= 1):
            return predicted_label
        return "chitchat"
    
    @classmethod
    def get_llm_response(cls, message, config={}):
        if not cls.is_simulation:
            response = cls.llm.invoke(message)
            print("LLM answers -- ", response.content, "--")
            return response
        
        response = {}
        logging.info(f"Calling LLM with: {config}")
        if config.get("mode") == "extract parameters":
            if config["intent"] in ["list_restaurants_tool", "list_cancellation_reasons_tool"]:
                pass
            elif config["intent"] == "find_customer_bookings_tool":
                response["date_range"] = ["2025-08-01", "2025-08-30"]
                response["party_size_range"] = [3, None]
            elif config["intent"] == "search_availability_tool":
                response["restaurant_name"] = "TheHungryUnicorn"
                response["visit_date"] = "2025-08-15"
                response["party_size"] = 4
            elif config["intent"] == "create_booking_tool":
                response["restaurant_name"] = "TheHungryUnicorn"
                response["visit_date"] = "2025-08-15"
                response["visit_time"] = "19:00:00"
                response["party_size"] = 4
                response["customer_email"] = "aborjigin@gmail.com"
            
        elif config.get("mode") == "extract update cancel":
            if config["intent"] == "update_booking_details_tool":
                response["new_party_size"] = 7
                response["restaurant_name"] = "TheHungryUnicorn" # For find_customer_bookings part
            elif config["intent"] == "cancel_booking_tool":
                response["restaurant_name"] = "TheHungryUnicorn" # For find_customer_bookings part
        elif config.get("mode") == "extract additional":
            # Simulate extracting specific missing parameters
            response = {
                "customer_first_name": "Altan",
                "customer_surname": "Borjigin",
                "customer_mobile": "112735089"
            }

        class SimulatedResponse:
            def __init__(self):
                self.content = json.dumps(response)
                if not response:
                    self.content = "No parameters"
        return SimulatedResponse()

tool_parameters_schema = {
    "list_restaurants_tool": {},
    "list_cancellation_reasons_tool": {},
    "find_customer_bookings_tool": {
        "date_range": "list of two ISO formatted dates (YYYY-MM-DD) for start and end, e.g., ['2025-01-01', '2025-01-31']. Can be partial like ['2025-01-01', null] or [null, '2025-01-31'].",
        "time_range": "list of two ISO formatted times (HH:MM:SS) for start and end, e.g., ['18:00:00', '20:00:00']. Can be partial.",
        "restaurant_name": "name of the restaurant (e.g., 'The Gourmet Palace').",
        "party_size_range": "list of two integers for minimum and maximum party size, e.g., [2, 4]. Can be partial.",
        "status": "status of the booking ('cancelled', 'confirmed', 'completed').",
        "is_leave_time_confirmed": "boolean indicating if leave time is confirmed (true/false).",
        "room_number": "room number if customer is staying at a hotel (string)."
    },
    "search_availability_tool": {
        "restaurant_name": "name of the restaurant (e.g., 'The Gourmet Palace').",
        "visit_date": "desired date for the visit in ISO format (YYYY-MM-DD).",
        "party_size": "number of people in the party (integer)."
    },
    "create_booking_tool": {    
        "restaurant_name": "name of the restaurant (e.g., 'The Gourmet Palace').",
        "visit_date": "date of the visit in ISO format (YYYY-MM-DD).",
        "visit_time": "time of the visit in ISO format (HH:MM:SS).",
        "party_size": "number of people for the reservation (integer).",
        "customer_email": "customer email address (e.g., 'john.doe@example.com').",
        "customer_first_name": "customer first name.",
        "customer_surname": "customer surname.",
        "customer_mobile": "customer mobile phone number.",
        "special_requests": "any special requests (e.g., 'window seat', 'allergies').",
        "is_leave_time_confirmed": "indicates if departure time is confirmed (true/false).",
        "room_number": "room number if applicable (string).",
    },
    "cancel_booking_tool": {    
        "microsite_name": "microsite name (often same as restaurant name).",
        "cancellation_reason_id": "ID of the cancellation reason (integer), get from list_cancellation_reasons_tool.",
    },
    "update_booking_details_tool": {    
        "new_visit_date": "updated date for the visit in ISO format (YYYY-MM-DD).",
        "new_visit_time": "updated time for the visit in ISO format (HH:MM:SS).",
        "new_party_size": "updated number of people for the reservation (integer).",
        "new_special_requests": "updated special requests (string).",
        "new_is_leave_time_confirmed": "updated status for whether leave time is confirmed (true/false)."
    }
}

user_required_parameters = {
    "find_customer_bookings_tool": [],
    "search_availability_tool": ["restaurant_name", "visit_date", "party_size"],
    "create_booking_tool": ["restaurant_name", "visit_date", "visit_time", "party_size", "customer_email", "customer_first_name", "customer_surname", "customer_mobile"],
    "cancel_booking_tool": ["cancellation_reason_id"],
    "update_booking_details_tool": ["new_visit_date", "new_visit_time", "new_party_size", "new_special_requests", "new_is_leave_time_confirmed"],
}

extraction_prompt_template = """
You are an expert at extracting information from natural language.
Extract the relevant input parameters for the tools `{tool_name}` from the user's query.
If a parameter is not explicitly mentioned or cannot be inferred, return null for that parameter.
Pay close attention to date and time formats (YYYY-MM-DD, HH:MM:SS).
For ranges (date_range, time_range, party_size_range), provide a list of two elements, [start, end]. If only one part of the range is specified, use null for the other part, e.g., ["2025-01-01", null]. Range is inclusive on both ends.

Expected Parameters: {parameters_description}

User Query: "{user_query}"

Provide the extracted parameters as a JSON object. Ensure numerical values are actual numbers, boolean values are true/false, and lists are properly formatted.
Example for date_range: ["2025-08-01", "2025-08-31"]
Example for party_size_range: [2, 4]
Example for single date: "2025-08-15"
Example for single time: "19:00:00"

JSON:
"""

json_parser = JsonOutputParser()

@dataclass
class AgentState:
    entry_point: str = ""
    user_message: Optional[str] = None
    current_intent: Optional[str] = None
    original_intent: Optional[str] = None
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    original_action_params: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    found_results: Optional[List[Dict[str, Any]]] = None
    selected_email: str = ""
    selected_booking_reference: Optional[int] = None
    selected_restaurant_name: Optional[str] = None
    selected_booking_index: Optional[int] = None
    agent_response: Optional[str] = None
    chat_history: List[Union[HumanMessage, AIMessage]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        if role == "user":
            self.chat_history.append(HumanMessage(content=content))
        elif role == "ai":
            self.chat_history.append(AIMessage(content=content))
        self.chat_history = self.chat_history[-10:]

    def get_context(self):
        context = ""
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                context += message.content + "; "
        return context
    
    def reset(self):
        self.entry_point = None
        self.user_message=None
        self.current_intent=None
        self.extracted_params={}
        self.original_action_params={}
        self.required_params=[]
        self.found_results=None
        self.selected_booking_reference=None
        self.selected_restaurant_name=None
        self.selected_booking_index=None
        self.agent_response=None
        self.original_intent=None
        self.chat_history=[AIMessage(content="Hello! I'm your restaurant booking assistant. I can help you check availability, make bookings, or manage existing reservations. What would you like to do?")]
        return self

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
        for key, value in content.items():
            self.tbody += f"<tr><th>{key}</th><td>{value}</td></tr>"
        return self.html()
    def html(self):
        return HTMLTableCreator.prefix + self.thead + self.tbody + HTMLTableCreator.postfix    

def classify_and_extract_initial_params(state: AgentState) -> AgentState:
    state.add_message("user", state.user_message)
    if state.entry_point:
        return state
    logging.info(f"--- Node: classify_and_extract_initial_params ---")
    
    user_query = state.user_message

    intent = AIModel.classify_intent(user_query)
    logging.info(f"Classified intent: {intent} [{user_query} ({len(state.chat_history)})]")
    # Reset specific state fields for a new turn, preserving history and selected_email
    state.reset()
    state.current_intent = intent

    if intent == "chitchat":
        state.agent_response = "<p>Hello, please provide more information of what you want me to help with.</p>"
        state.add_message("ai", state.agent_response)
        return state

    if intent not in tool_parameters_schema:
        state.agent_response = "<p>I'm sorry, I don't understand that request. Please try rephrasing.</p>"
        state.add_message("ai", state.agent_response)
        return state

    if intent in ["update_booking_details_tool", "cancel_booking_tool"]:
        state.original_intent = intent
        state.current_intent = "find_customer_bookings_tool"
        
        combined_params_schema = {
            **tool_parameters_schema["find_customer_bookings_tool"],
            **tool_parameters_schema[intent]
        }
        find_params_desc = json.dumps(combined_params_schema)

        extraction_prompt_find = extraction_prompt_template.format(
            tool_name=f"{state.current_intent} and {intent}",
            parameters_description=find_params_desc,
            user_query=user_query
        )
        try:
            llm_response = AIModel.get_llm_response(extraction_prompt_find, {"intent": intent, "mode": "extract update cancel"})
            extracted = json_parser.parse(llm_response.content)
            
            for key, val in extracted.items():
                if key in tool_parameters_schema["find_customer_bookings_tool"]:
                    state.extracted_params[key] = val
                elif key in tool_parameters_schema[intent]:
                    state.original_action_params[key] = val
            logging.info(f"Extracted params for find_customer_bookings_tool and {intent}: {extracted}")
        except Exception as e:
            logging.error(f"Error extracting find_customer_bookings_tool params: {e}")
            state.agent_response = "<p>I had trouble extracting details for finding your booking. Could you please provide your email address?</p>"
            state.add_message("ai", state.agent_response)
            state.required_params = ["email"]
            return state
        
    else:        
        params_desc = json.dumps(tool_parameters_schema[intent])
        extraction_prompt = extraction_prompt_template.format(
            tool_name=intent,
            parameters_description=params_desc,
            user_query=user_query
        )
        try:
            if tool_parameters_schema[intent]:
                llm_response = AIModel.get_llm_response(extraction_prompt, {"intent":intent, "mode": "extract parameters"})
                extracted = json_parser.parse(llm_response.content)
                state.extracted_params.update(extracted)
                logging.info(f"Extracted params for {intent}: {extracted}")
            else:
                logging.info(f"No parameters to extract for {intent}.")
        except Exception as e:
            logging.error(f"Error extracting params for {intent}: {e}")
            state.agent_response = "<p>I had trouble understanding some details. Can you rephrase or provide the details explicitly?</p>"
            state.add_message("ai", state.agent_response)
            return state

    return state

update_missing_template = "new_visit_date, new_visit_time, new_party_size (number of people), new_special_requests, or whether leave time is confirmed (true/false)"
def check_and_prompt_for_missing_params(state: AgentState) -> AgentState:
    if state.entry_point and not state.entry_point == "check_and_prompt_for_missing_params":
        return state
    logging.info(f"--- Node: check_and_prompt_for_missing_params ---")
    
    missing = []
    current_params = state.extracted_params
    logging.info(f"Current params: {current_params}; original action params: {state.original_action_params}")
    
    if state.original_intent == "update_booking_details_tool":
        if not any(state.original_action_params.get(p) is not None for p in tool_parameters_schema["update_booking_details_tool"]):
            missing = [update_missing_template]
    elif state.original_intent == "cancel_booking_tool" and not isinstance(state.original_action_params.get("cancellation_reason_id"), int):
        missing = ["cancellation_reason_id"]
    else:
        required_for_current_intent = user_required_parameters.get(state.current_intent, [])
        for param in required_for_current_intent:
            if current_params.get(param) is None or (isinstance(current_params.get(param), list) and all(x is None for x in current_params.get(param))):
                missing.append(param)
        
        if state.current_intent == "find_customer_bookings_tool" and not state.selected_email:
            if "email" not in missing:
                missing.append("email")

    state.required_params = list(set(missing))
    logging.info(f"Missing parameters: {state.required_params}")

    if state.required_params:
        if state.agent_response and "trouble" in state.agent_response:
            pass
        else:
            param_names_friendly = [p.replace('_', ' ') for p in state.required_params]
            if "email" in state.required_params and state.selected_email == "":
                state.agent_response = "<p>I need your email address to find your bookings. Please provide it.</p>"
            elif len(param_names_friendly) == 1:
                state.agent_response = f"<p>I need the {param_names_friendly[0]} to proceed. Please provide it.</p>"
            elif len(param_names_friendly) > 1:
                state.agent_response = f"<p>I need the following information: {', '.join(param_names_friendly)}. Please provide them.</p>"
            else:
                state.agent_response = "<p>I need more details to complete your request. What information can you provide?</p>"
            if state.original_intent == "cancel_booking_tool":
                reasons = tools.list_cancellation_reasons_tool()
                state.found_results = reasons
                columns = ["ID", "Title", "Description"]
                table = HTMLTableCreator(columns)
                for reason in reasons:
                    table.add_row([reason.get('id'), reason.get('reason'), reason.get('description')])
                state.agent_response += table.html()

        state.entry_point = "extract_additional_pause"
        state.add_message("ai", state.agent_response)
        return state
    state.entry_point = "execute_tool_action"
    return state

def extract_additional_params(state: AgentState) -> AgentState:
    logging.info(f"--- Node: extract_additional_params ---")
    state.entry_point = "check_and_prompt_for_missing_params"
    user_query = state.user_message

    if state.original_intent == "cancel_booking_tool":
        word = re.search(r"\b\d+", user_query)
        if word:
            num = int(word.group())
            if any("id" in reason for reason in state.found_results):
                state.original_action_params["cancellation_reason_id"] = int(num)
                state.agent_response = f"<p>Confirmed cancellation reason {word}</p>"
                state.found_results = None
                return state
        state.agent_response = "<p>Not a number, or not a valid cancellation ID. Please enter again. </p>"
        return state
    
    missing_params_to_extract = {}
    
    for p in state.required_params:
        if p == "email":
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_query)
            if email_match:
                state.selected_email = email_match.group(0)
                logging.info(f"Extracted email: {state.selected_email}")
                continue
        
        if p in tool_parameters_schema.get(state.current_intent, {}):
            missing_params_to_extract[p] = tool_parameters_schema[state.current_intent][p]
        elif state.original_intent and p in tool_parameters_schema.get(state.original_intent, {}):
            missing_params_to_extract[p] = tool_parameters_schema[state.original_intent][p]
    
    if update_missing_template in state.required_params:
        for p in ["new_visit_date", "new_visit_time", "new_party_size", "new_special_requests", "new_is_leave_time_confirmed"]:
             if p not in missing_params_to_extract and state.original_intent == "update_booking_details_tool":
                missing_params_to_extract[p] = tool_parameters_schema[state.original_intent][p]

    if not missing_params_to_extract:
        return state

    extraction_prompt = extraction_prompt_template.format(
        tool_name=f"current task (extracting for {state.current_intent} / {state.original_intent or ''})",
        parameters_description=json.dumps(missing_params_to_extract),
        user_query=user_query
    )
    
    try:
        llm_response = AIModel.get_llm_response(extraction_prompt, {"mode": "extract additional"})
        extracted = json_parser.parse(llm_response.content)
        for param, value in extracted.items():
            if value is not None and value != []:
                if param in tool_parameters_schema.get(state.current_intent, {}):
                    state.extracted_params[param] = value
                elif state.original_intent and param in tool_parameters_schema.get(state.original_intent, {}):
                    state.original_action_params[param] = value
        
        logging.info(f"Updated extracted params: {state.extracted_params}")
        logging.info(f"Updated original action params: {state.original_action_params}")
        
        return state

    except Exception as e:
        logging.error(f"Error extracting additional params: {e}")
        state.agent_response = "<p>I had trouble understanding that. Could you please provide the information more clearly?</p>"
        state.add_message("ai", state.agent_response)
        state.required_params = ["clarification"]
        return state

def execute_tool_action(state: AgentState) -> AgentState:
    if state.entry_point != "execute_tool_action":
        return state
    logging.info(f"--- Node: execute_tool_action ---")
    tool_name = state.current_intent
    
    if state.original_intent == state.current_intent:
        params_for_tool = {**state.extracted_params, **state.original_action_params}
    else:
        params_for_tool = state.extracted_params

    final_params = {k: v for k, v in params_for_tool.items() if v is not None}
    
    if "new_party_size" in final_params and isinstance(final_params["new_party_size"], str):
        try: final_params["new_party_size"] = int(final_params["new_party_size"])
        except ValueError: final_params.pop("new_party_size")
    if "party_size" in final_params and isinstance(final_params["party_size"], str):
        try: final_params["party_size"] = int(final_params["party_size"])
        except ValueError: final_params.pop("party_size")
    if "party_size_range" in final_params and isinstance(final_params["party_size_range"], list):
        try: final_params["party_size_range"] = [int(p) if p is not None else None for p in final_params["party_size_range"]]
        except (ValueError, TypeError): final_params.pop("party_size_range")
    if "cancellation_reason_id" in final_params and isinstance(final_params["cancellation_reason_id"], str):
        try: final_params["cancellation_reason_id"] = int(final_params["cancellation_reason_id"])
        except ValueError: final_params.pop("cancellation_reason_id")
    if "is_leave_time_confirmed" in final_params and isinstance(final_params["is_leave_time_confirmed"], str):
        final_params["is_leave_time_confirmed"] = final_params["is_leave_time_confirmed"].lower() == 'true'
    if "new_is_leave_time_confirmed" in final_params and isinstance(final_params["new_is_leave_time_confirmed"], str):
        final_params["new_is_leave_time_confirmed"] = final_params["new_is_leave_time_confirmed"].lower() == 'true'

    logging.info(f"Attempting to call tool: {tool_name} with final params: {final_params}")

    try:
        tool_function = getattr(tools, tool_name)
        sig = inspect.signature(tool_function)
        
        callable_params = {k: v for k, v in final_params.items() if k in sig.parameters}
        
        if "email" in sig.parameters and state.selected_email:
            callable_params["email"] = state.selected_email
        if state.selected_booking_reference is not None and "booking_reference" in sig.parameters:
            callable_params["booking_reference"] = state.selected_booking_reference
        if state.selected_restaurant_name is not None and "restaurant_name" in sig.parameters:
            callable_params["restaurant_name"] = state.selected_restaurant_name
        
        if tool_name == "cancel_booking_tool" and "microsite_name" in sig.parameters and state.selected_restaurant_name:
            callable_params["microsite_name"] = state.selected_restaurant_name

        result = tool_function(**callable_params)
        logging.info(f"Tool call result: {len(result)}")

        if tool_name == "find_customer_bookings_tool":
            state.found_results = result
            if state.original_intent:
                state.entry_point = "handle_found_bookings"
                return state
            else:
                if not result:
                    state.agent_response = "<p>No bookings found matching your criteria.</p>"
                else:
                    columns = ["#", "Reference", "Restaurant", "Date", "Time", "Party Size", "Status"]
                    table = HTMLTableCreator(columns)
                    for i, booking in enumerate(result):
                        table.add_row([i+1, booking.get('booking_reference'), booking.get('restaurant'), 
                                     booking.get('visit_date'), booking.get('visit_time'), 
                                     booking.get('party_size'), booking.get('status')])
                    state.agent_response = f"<p>Here are the bookings I found:</p>{table.html()}"
        elif tool_name == "search_availability_tool":
            if result and result.get("available_slots"):
                slots = result["available_slots"]
                if slots:
                    columns = ["Time", "Available", "Max Party Size"]
                    table = HTMLTableCreator(columns)
                    for slot in slots:
                        table.add_row([slot.get('time'), 'Yes' if slot.get('available') else 'No', slot.get('max_party_size')])
                    state.agent_response = f"<p>I found availability at {final_params.get('restaurant_name', 'the restaurant')} on {final_params.get('visit_date', 'the date you specified')} for {final_params.get('party_size', 'your party size')}:</p>{table.html()}"
                else:
                     state.agent_response = f"<p>No availability found for {final_params.get('restaurant_name', 'the restaurant')} on {final_params.get('visit_date', 'the date you specified')} for {final_params.get('party_size', 'your party size')}. Please try a different date or party size.</p>"
            else:
                state.agent_response = f"<p>No availability found for {final_params.get('restaurant_name', 'the restaurant')} on {final_params.get('visit_date', 'the date you specified')} for {final_params.get('party_size', 'your party size')}. Please try a different date or party size.</p>"
        elif tool_name == "create_booking_tool":
            if result and result.get("booking_reference"):
                state.agent_response = f"<p>Your booking at <i>{final_params.get('restaurant_name', 'the restaurant')}</i> on {final_params.get('visit_date', 'your specified date')} - {final_params.get('visit_time', 'your specified time')} for {final_params.get('party_size', 'your party size')} people is confirmed. Your reference is: <b>{result['booking_reference']}</b>.</p>"
            else:
                state.agent_response = f"<p>Failed to create booking. Error: {result.get('error', 'API response error.')}</p>"
        elif tool_name == "cancel_booking_tool":
            if result and result.get("status") == "cancelled":
                state.agent_response = f"<p>Your booking <b>{state.selected_booking_reference}</b> at {final_params.get('restaurant_name', 'the restaurant')} has been successfully cancelled.</p>"
            else:
                state.agent_response = f"<p>Failed to cancel booking {state.selected_booking_reference}. Error: {result.get('error', 'Unknown error.')}</p>"
        elif tool_name == "update_booking_details_tool":
            update_table = HTMLTableCreator(["Field", "New value"])
            update_summary = update_table.from_dict({k.replace('_', ' '): v for k, v in state.original_action_params.items()})
            if result and result.get("status") == "updated":
                state.agent_response = f"<p>Booking <b>{state.selected_booking_reference}</b> at <i>{final_params.get('restaurant_name', 'the restaurant')}</i> has been successfully updated with the folling: {update_summary} </p>"
            else:
                state.agent_response = f"<p>Calling API to update <b>{state.selected_booking_reference}</b> at <i>{final_params.get('restaurant_name', 'the restaurant')}</i> with the folling, but no changes have been made. {update_summary} </p>"

        elif tool_name == "list_restaurants_tool":
            if result:
                columns = ["ID", "Name", "Microsite Name"]
                table = HTMLTableCreator(columns)
                for restaurant in result:
                    table.add_row([restaurant.get('id'), restaurant.get('name'), restaurant.get('microsite_name')])
                state.agent_response = f"<p>Here are the restaurants:</p>{table.html()}"
            else:
                state.agent_response = "<p>No restaurants are currently registered.</p>"
        elif tool_name == "list_cancellation_reasons_tool":
            if result:
                columns = ["ID", "Title", "Description"]
                table = HTMLTableCreator(columns)
                for reason in result:
                    table.add_row([reason.get('id'), reason.get('reason'), reason.get('description')])
                state.agent_response = f"<p>Here are the cancellation reasons:</p>{table.html()}"
            else:
                state.agent_response = "<p>No cancellation reasons found.</p>"

    except Exception as e:
        logging.exception(f"Error calling tool {tool_name}:")
        state.agent_response = f"<p>An error occurred while trying to process your request using the {tool_name} tool: {str(e)}. Please try again.</p>"

    state.add_message("ai", state.agent_response)
    return state

def handle_found_bookings(state: AgentState) -> AgentState:
    if state.entry_point != "handle_found_bookings":
        return state
    logging.info(f"--- Node: handle_found_bookings ---")
    if not state.found_results:
        state.agent_response = "<p>I couldn't find any bookings matching your criteria. If you'd like to continue, try describe your booking with new parameters; Otherwise, reset the chat.</p>"
        state.entry_point = "check_and_prompt_for_missing_params"
        state.add_message("ai", state.agent_response)
    else:
        logging.info(f"Found {len(state.found_results)} bookings")
        columns = ["#", "Reference", "Restaurant", "Date", "Time", "Party Size", "Status"]
        table = HTMLTableCreator(columns)
        for i, booking in enumerate(state.found_results):
            table.add_row([i+1, booking.get('booking_reference'), booking.get('restaurant'),
                         booking.get('visit_date'), booking.get('visit_time'),
                         booking.get('party_size'), booking.get('status')])
        state.agent_response = f"<p>I found multiple bookings. Please select the one you'd like to update/cancel by typing its number:</p>{table.html()}"
        state.add_message("ai", state.agent_response)
        state.entry_point = "select_booking_by_index"
    return state

def select_booking_by_index(state: AgentState) -> AgentState:
    logging.info(f"--- Node: select_booking_by_index ---")
    user_input = state.user_message
    num = re.search(r"\b\d+", user_input)

    if not num:
        state.agent_response = "<p>That's not a valid number. Please enter the number corresponding to the booking you want to select.</p>"
        state.add_message("ai", state.agent_response)
        return state
    try:
        index = int(num.group()) - 1
        if 0 <= index < len(state.found_results):
            state.selected_booking_index = index
            state.selected_booking_reference = state.found_results[index]['booking_reference']
            state.selected_restaurant_name = state.found_results[index]['restaurant']
            logging.info(f"Booking selected: {state.selected_booking_reference}: {state.selected_restaurant_name}")
            state.agent_response = f"<p>You selected booking reference {state.selected_booking_reference}. Proceeding with your request with restaurant {state.selected_restaurant_name}.</p>"
            state.add_message("ai", state.agent_response)
            state.entry_point = "execute_tool_action"
            state.current_intent = state.original_intent
        else:
            state.agent_response = f"<p>Invalid selection. Please choose a number between 1 and {len(state.found_results)}.</p>"
            state.add_message("ai", state.agent_response)
    except (ValueError, IndexError):
        state.agent_response = "<p>Invalid selection. Please enter a valid number.</p>"
        state.add_message("ai", state.agent_response)
    
    return state

def pause_or_exit(state: AgentState):
    logging.info(f"<Pause workflow for more user information or final output, entry point {state.entry_point}>")
    return state

# LangGraph Workflow Definition
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("classify_and_extract", classify_and_extract_initial_params)
workflow.add_node("check_and_prompt", check_and_prompt_for_missing_params)
workflow.add_node("extract_additional", extract_additional_params)
workflow.add_node("execute_tool", execute_tool_action)
workflow.add_node("handle_found_bookings", handle_found_bookings)
workflow.add_node("select_booking", select_booking_by_index)
workflow.add_node("pause_or_exit", pause_or_exit)

# Set entry point
workflow.set_entry_point("classify_and_extract")

# Define edges
workflow.add_conditional_edges(
    "classify_and_extract",
    lambda state: state.current_intent == "chitchat",
    {
        True: "pause_or_exit",
        False: "check_and_prompt"
    } 
)

def route_check_and_prompt(state: AgentState):
    if state.entry_point == "extract_additional_resume":
        return "extract_additional"
    elif state.current_intent == "chitchat" or state.required_params:
        return "pause_or_exit"
    else:
        return "execute_tool"

workflow.add_conditional_edges(
    "check_and_prompt",
    route_check_and_prompt,
    {
        "execute_tool": "execute_tool",
        "extract_additional": "extract_additional",
        "pause_or_exit": "pause_or_exit"
    }
)

workflow.add_edge("extract_additional", "check_and_prompt")

def route_after_tool_execution(state: AgentState):
    if state.entry_point == "select_booking_by_index":
        return "select_booking"
    if state.entry_point == "handle_found_bookings":
        return "handle_found_bookings"
    if state.current_intent == "find_customer_bookings_tool" and state.original_intent and state.current_intent != state.original_intent:
        if state.found_results and state.selected_booking_index is not None:
            state.current_intent = state.original_intent
            return "execute_tool"
    return "pause_or_exit"

workflow.add_conditional_edges(
    "execute_tool",
    route_after_tool_execution,
    {
        "select_booking": "select_booking",
        "execute_tool": "execute_tool",
        "handle_found_bookings": "handle_found_bookings",
        "pause_or_exit": "pause_or_exit"
    }
)

workflow.add_edge("handle_found_bookings", "pause_or_exit")

workflow.add_conditional_edges(
    "select_booking", 
    lambda state: state.entry_point == "execute_tool_action",
    {
        True: "execute_tool",
        False: "pause_or_exit"
    }
)

# NEW: Initialize the checkpointer
_memory_saver_context = SqliteSaver.from_conn_string(":memory:")
memory = _memory_saver_context.__enter__()

# Compile the graph with the checkpointer
app = workflow.compile(checkpointer=memory)

# Initial state for a new conversation.
# This serves as the default state for new threads when the checkpointer is used.
initial_agent_state = AgentState()

# Interactive Chat Loop (for local testing)
def run_agent():
    logging.info("Agent started. Type 'exit' to quit.")
    print("Hello! I can help you with restaurant bookings. How can I assist you?")

    test_session_id = "cli_session" 
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        config = {"configurable": {"thread_id": test_session_id}}

        try:
            # Get the current state. If it's a new thread, it will be None.
            # The LangGraph `stream` method with a checkpointer will initialize it
            # implicitly using the `AgentState` schema and potentially an initial snapshot.
            
            # For the first message, `initial_agent_state`'s history is used.
            # Subsequent messages will append to the history loaded by the checkpointer.

            # Prepare inputs for the current turn. These are arguments to the entry point.
            inputs_for_this_turn = {
                "user_message": user_input,
                "selected_email": "" # Default for CLI, or could be passed as CLI arg
            }

            # Run the stream. The checkpointer handles loading/saving state.
            for s_step in app.stream(inputs_for_this_turn, config=config):
                # We log the raw step output, but don't manually merge state here
                logging.debug(f"Stream step yielded: {s_step}")
                if "__end__" in s_step: # LangGraph signals termination with '__end__' key
                    logging.info("Graph stream reached END.")
                    break

            # After the stream, retrieve the *final* state from the checkpointer
            final_checkpoint = app.get_state(config)
            if final_checkpoint:
                final_state = AgentState(**final_checkpoint.values)
                print(f"AI: {final_state.agent_response}")
            else:
                logging.error(f"Could not retrieve final state for session {test_session_id}.")
                print("AI: I'm not sure how to proceed or encountered an issue. Can you rephrase?")
            
        except Exception as e:
            logging.exception("An error occurred during agent execution:")
            print(f"AI: I encountered an error: {str(e)}. Please try again.")
            try:
                # Delete the thread state on error to allow a fresh start
                app.delete_thread(config)
                logging.info(f"Deleted conversation state for session {test_session_id} due to error.")
            except Exception as delete_e:
                logging.error(f"Error deleting thread {test_session_id}: {delete_e}")

if __name__ == "__main__":
    pass
    # run_agent()
import os
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import app.ai_tools as tools


os.environ["GOOGLE_API_KEY"] = ""
llm = init_chat_model("google_genai:gemini-2.0-flash-lite")

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([tools.customer_bookings_and_restaurants_summary, ])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("Tell me about the customer with the email 'yingyan797@restaurantai.com'")

# Get the tool call
print(msg)
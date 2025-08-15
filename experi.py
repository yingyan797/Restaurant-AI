import os
import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SimpleState:
    message: str = ""
    response: str = ""
    history: list = field(default_factory=list)
    # New flag to indicate termination
    should_terminate: bool = False

def entry_node(state: SimpleState) -> SimpleState:
    logging.info(f"NODE_ENTRY: entry_node. Message: {state.message}")
    state.response = f"Processed: {state.message}"
    state.history.append(HumanMessage(content=state.message))
    logging.info(f"NODE_EXIT: entry_node. Response: {state.response}")
    return state

# This node now only updates the state, it does not directly return a routing decision
def terminate_decision_node(state: SimpleState) -> SimpleState:
    logging.info(f"NODE_ENTRY: terminate_decision_node. Response: {state.response}")
    # In a real scenario, this node would decide based on complex logic
    state.should_terminate = True # For this test, we always want to terminate
    logging.info(f"NODE_EXIT: terminate_decision_node. Set should_terminate={state.should_terminate}")
    return state

# This is the dedicated routing function that reads the state
def decide_next_step(state: SimpleState) -> str:
    logging.info(f"ROUTER_FUNC_EVAL: decide_next_step. should_terminate: {state.should_terminate}")
    if state.should_terminate:
        logging.info("ROUTER_FUNC_DECISION: decide_next_step -> END")
        return END # Return the END object to signal termination
    else:
        # This branch won't be hit in this specific test, but useful for real graphs
        logging.info("ROUTER_FUNC_DECISION: decide_next_step -> 'continue_processing'")
        return "continue_processing" # A dummy node name if not terminating

# Build the simple graph
workflow = StateGraph(SimpleState)
workflow.add_node("entry", entry_node)
workflow.add_node("terminate_decision", terminate_decision_node)

workflow.set_entry_point("entry")
workflow.add_edge("entry", "terminate_decision") # Go from entry to the decision node

# Connect the decision node to the conditional edge, using the routing function
workflow.add_conditional_edges(
    "terminate_decision", # The node whose output triggers this routing
    decide_next_step,     # The function that decides the next step (reads state)
    {
        END: END,         # If decide_next_step returns END, go to END
        "continue_processing": "entry" # Dummy path if not terminating, for graph completeness
    }
)

# Compile the graph
app_simple = workflow.compile()

# --- Test Execution ---
if __name__ == "__main__":
    print("Running simple LangGraph END test...")
    initial_state = SimpleState(message="Hello, test!", should_terminate=False)

    final_state_snapshot = None
    end_key_detected = False

    try:
        for s_step in app_simple.stream(initial_state.__dict__):
            logging.info(f"STREAM YIELDED: Keys: {list(s_step.keys())}, Full Step: {s_step}")

            # Apply updates to the state for demonstration
            # Filter out special LangGraph internal keys like '__end__' when updating attributes
            for k, v in s_step.items():
                if hasattr(initial_state, k) and k != "__end__": # Ensure not trying to set __end__ as attribute
                    setattr(initial_state, k, v)
            
            # Keep track of the latest state snapshot
            final_state_snapshot = initial_state

            if "__end__" in s_step:
                logging.info(f"SUCCESS: __end__ key DETECTED in stream! Final response (from state): {initial_state.response}")
                end_key_detected = True
                break

            logging.info(f"Stream step processed. Current response (from state): {initial_state.response}")

    except Exception as e:
        logging.error(f"Error during stream execution: {e}", exc_info=True)

    if not end_key_detected:
        print("\nFAILURE: __end__ key was NOT detected in the stream.")
        if final_state_snapshot:
            print(f"Last state snapshot: {final_state_snapshot}")
    else:
        print("\nSUCCESS: The __end__ key was detected as expected.")

    print("\nTest finished.")
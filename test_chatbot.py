#!/usr/bin/env python3
"""
Simple test script to verify the chatbot integration works.
"""

import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_chatbot_import():
    """Test if we can import the chatbot components."""
    try:
        from app.routers.chatbot import router
        print("✓ Chatbot router imported successfully")
        
        from app.ai_chatbot import graph, initial_agent_state
        print("✓ AI chatbot components imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic chatbot functionality."""
    try:
        from app.ai_chatbot import graph, initial_agent_state
        from langchain_core.messages import HumanMessage
        import copy
        
        # Create a test state
        test_state = copy.deepcopy(initial_agent_state)
        test_state['messages'].append(HumanMessage(content="Hello, I want to check availability"))
        
        # Try to process one step
        for state in graph.stream(test_state):
            print("✓ Graph processing works")
            break
            
        return True
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Restaurant AI Chatbot Integration...")
    print("=" * 50)
    
    if test_chatbot_import():
        if test_basic_functionality():
            print("\n✓ All tests passed! Chatbot integration is ready.")
        else:
            print("\n✗ Functionality test failed.")
    else:
        print("\n✗ Import test failed.")
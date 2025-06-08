"""
Main application for the MedicalCare AI assistant system.
"""

from flask import Flask, request, jsonify, session
from langgraph.types import Command
from langchain_core.messages import AIMessage
import uuid
import logging
from agent_setup import initialize_agent
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='medical_care.log'
)
logger = logging.getLogger('main')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize the agent
try:
    graph, memory = initialize_agent()
    logger.info("Agent loaded successfully")
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}")
    logger.critical(traceback.format_exc())
    graph, memory = None, None

@app.route('/chat', methods=['POST'])
def chat_bot():
    """Endpoint for chat interactions with the medical assistant."""
    if not graph:
        return jsonify({
            "error": "Agent not initialized",
            "responses": ["I'm sorry, the system is currently unavailable. Please try again later."]
        }), 500
    
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        user_id = data.get("user_id", str(uuid.uuid4()))
        
        # Check if this is a continuing conversation
        if not session.get('thread_id'):
            session['thread_id'] = str(uuid.uuid4())
        thread_id = session.get('thread_id')
        
        logger.info(f"Chat request from user {user_id}, thread {thread_id}")
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        human_command = Command(update={"messages": user_input})
        assistant_responses = []
        
        for output in graph.stream(human_command, config=config, stream_mode="values"):
            messages = output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    assistant_responses.append(msg.content)
        
        return jsonify({
            "responses": assistant_responses,
            "thread_id": thread_id
        })
        
    except Exception as e:
        logger.exception("Error processing chat request")
        return jsonify({
            "error": str(e),
            "responses": ["I'm sorry, an error occurred while processing your request."]
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation thread."""
    session.pop('thread_id', None)
    return jsonify({"status": "success", "message": "Conversation reset"})

# Run the app if executed directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

"""
Sets up the LLM and agent for the MedicalCare system.
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import logging
from config import SYSTEM_PROMPT, LLM_REPO_ID, LLM_MAX_TOKENS, LLM_DO_SAMPLE
from api_tools import available_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='medical_care.log'
)
logger = logging.getLogger('agent_setup')

def initialize_agent():
    """Initialize and return the agent with all required components."""
    
    logger.info("Initializing AI agent")
    
    # Create a memory saver for checkpointing
    memory = MemorySaver()
    
    try:
        # Initialize the language model
        llm = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            task="text-generation",
            max_new_tokens=LLM_MAX_TOKENS,
            do_sample=LLM_DO_SAMPLE,
        )
        
        # Create chat model
        chat = ChatHuggingFace(llm=llm, verbose=True)
        
        # Create the agent graph
        graph = create_react_agent(
            chat,
            tools=available_tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=memory,
        )
        
        logger.info("Agent initialized successfully")
        return graph, memory
        
    except Exception as e:
        logger.exception("Failed to initialize agent")
        raise e

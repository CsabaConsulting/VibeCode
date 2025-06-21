from typing import List, Optional, Union, Dict, Any
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReActAgent:
    def __init__(self, tools: List[BaseTool], model: str = "gemini-1.5-pro"):
        """
        Initialize the ReAct agent using LangGraph with Google's Gemini.
        
        Args:
            tools: List of tools the agent can use
            model: Google Gemini model to use (default: gemini-1.5-pro)
        """
        # Initialize Gemini model with specific parameters for better performance
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,  # Slightly higher temperature for more creative responses
            max_output_tokens=2048,  # Limit response length
            top_p=0.95,  # Controls diversity of responses
            top_k=40,  # Controls diversity of responses
            convert_system_message_to_human=True  # Gemini doesn't support system messages directly
        )
        
        self.tools = tools
        self.conversation_history: List[BaseMessage] = []
        
        # Create the agent with the tools and LLM
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        # Create the agent executor with better error handling
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent long-running agent loops
            early_stopping_method="generate"  # Better handling of tool calls
        )
    
    def _format_chat_history(self, history: List[BaseMessage]) -> List[Union[HumanMessage, AIMessage]]:
        """Format chat history for the agent."""
        formatted = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted.append(HumanMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                formatted.append(AIMessage(content=msg.content))
        return formatted
        
    def run(self, user_query: str) -> str:
        """
        Run the agent with the given query.
        
        Args:
            user_query: The user's query
            
        Returns:
            str: The final response
        """
        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_query))
        
        try:
            # Format chat history for context
            formatted_history = "\n".join(
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                for msg in self.conversation_history[-5:]  # Use last 5 messages for context
            )
            
            # Prepare the input with history
            input_with_history = f"""Current conversation history:
            {formatted_history}
            
            Question: {user_query}
            """
            
            # Run the agent with the formatted input
            response = self.agent_executor.run(input_with_history)
            
            # Get the response content
            if not response:
                response_content = "I'm sorry, I couldn't process that request."
            elif isinstance(response, str):
                response_content = response
            else:
                response_content = str(response)
            
            # Add assistant's response to history
            self.conversation_history.append(AIMessage(content=response_content))
            
            return response_content
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"Error in agent execution: {error_msg}")
            self.conversation_history.append(AIMessage(content=error_msg))
            return error_msg

from typing import Dict, List, Any, Callable, Optional, Type, Union, Sequence
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
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
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.2,
            convert_system_message_to_human=True  # Gemini doesn't support system messages directly
        )
        
        self.tools = tools
        self.conversation_history: List[BaseMessage] = []
        
        # Create the agent with the tools and LLM
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=(
                "You are a helpful assistant that can use tools to answer questions. "
                "Always provide clear and concise responses."
            )
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
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
            # Format chat history for the agent
            formatted_history = self._format_chat_history(self.conversation_history[:-1])
            
            # Run the agent
            response = self.agent_executor.invoke({
                "input": user_query,
                "chat_history": formatted_history
            })
            
            # Get the response content
            response_content = response.get("output", "I'm sorry, I couldn't process that request.")
            
            # Add assistant's response to history
            self.conversation_history.append(AIMessage(content=response_content))
            
            return response_content
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.conversation_history.append(AIMessage(content=error_msg))
            return error_msg

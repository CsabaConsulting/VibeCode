from typing import Dict, List, Any, Callable, Optional, Type
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReActAgent:
    def __init__(self, tools: List[BaseTool], model: str = "gpt-4"):
        """
        Initialize the ReAct agent using LangGraph.
        
        Args:
            tools: List of tools the agent can use
            model: OpenAI model to use
        """
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.tools = tools
        self.conversation_history = []
        
        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can use tools to answer questions."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
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
            # Run the agent
            response = self.agent.invoke({
                "input": user_query,
                "chat_history": self.conversation_history[:-1],  # Exclude the current message
                "agent_scratchpad": []
            })
            
            # Add assistant's response to history
            self.conversation_history.append(AIMessage(content=response["output"]))
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.conversation_history.append(AIMessage(content=error_msg))
            return error_msg

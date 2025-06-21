import streamlit as st
from agent import ReActAgent
from tools.weather_tool import WeatherTool
from tools.search_tool import SearchTool
import os

# Set page config
st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize agent in session state
if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar for API key input
with st.sidebar:
    st.title("🔧 Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    if openai_api_key and (st.session_state.agent is None or openai_api_key != os.environ.get("OPENAI_API_KEY")):
        try:
            # Set the API key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
            # Initialize the agent with tools
            tools = [
                SearchTool(),
                WeatherTool()
            ]
            st.session_state.agent = ReActAgent(tools=tools)
            st.success("✅ Agent initialized successfully!")
            
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
    
    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "This is an agentic assistant powered by LangGraph's ReAct agent. "
        "It can search the web and check the weather."
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 Agentic Assistant")
st.caption("I can help you find information and check the weather!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif st.session_state.agent is None:
            st.error("Agent not initialized. Please check your API key and try again.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Get response from agent
                    response = st.session_state.agent.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add some styling
st.markdown("""
<style>
    .stTextInput input {
        height: 60px;
        font-size: 16px;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
    }
    .stAlert {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

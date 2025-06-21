import streamlit as st
from agent import ReActAgent
from tools.weather_tool import WeatherTool
from tools.search_tool import SearchTool
from tools.currency_tool import CurrencyConversionTool
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
    
    # Google AI Studio API Key input
    google_api_key = st.text_input("Google AI Studio API Key", type="password")
    
    if google_api_key and (st.session_state.agent is None or google_api_key != os.environ.get("GOOGLE_API_KEY")):
        try:
            # Set the API key for Google's Generative AI
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
            # Initialize the agent with tools
            tools = [
                SearchTool(),
                WeatherTool(),
                CurrencyConversionTool()
            ]
            st.session_state.agent = ReActAgent(tools=tools)
            st.success("✅ Agent initialized successfully with Gemini!")
            
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
    
    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "This is an agentic assistant powered by Google's Gemini model. "
        "It can search the web and check the weather using tools."
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
        if not google_api_key:
            st.error("Please enter your Google AI Studio API key in the sidebar.")
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

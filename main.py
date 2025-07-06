import streamlit as st
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner, function_tool
from dotenv import load_dotenv
import os
import requests
import asyncio

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Setup Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# Model and config
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
config = RunConfig(model=model, model_provider=client, tracing_disabled=True)

# Tool: Wikipedia-based disease summary
@function_tool
def get_health_info(disease: str) -> str:
    """Fetch disease summary from Wikipedia."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{disease}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "No summary found.")
        else:
            return f"No information found for '{disease}'."
    except Exception as e:
        return f"Error: {str(e)}"

# Agent
agent = Agent(
    name="Health Care Agent",
    instructions="You are a health advisor. Use the 'get_health_info' tool to explain any disease in simple terms.if someone told you diseas name in other language then translate it into english first and you can answare with your intelligents if someone ask anything else.",
    tools=[get_health_info]
)

# Streamlit UI
st.set_page_config(page_title="HealthCare Chat", page_icon="💬")
st.title("🤖 HealthCare Chatbot")
st.markdown("Ask any medical-related question and get a response!")

st.markdown("**Made by Uzair 💻**", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input and agent logic
if prompt := st.chat_input("Type your question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for assistant message
    with st.chat_message("assistant"):
        response_area = st.empty()
        
        try:
            # Run agent with proper async handling
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = Runner.run_sync(
                    agent,
                    input=prompt,
                    run_config=config
                )
            finally:
                loop.close()
            
            # Extract the actual response text
            response_text = response.final_output if hasattr(response, 'final_output') else str(response)
            
            # Show response
            response_area.markdown(response_text)
            
            # Add assistant message to history (store the text, not the response object)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            response_area.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Optional: Add a clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
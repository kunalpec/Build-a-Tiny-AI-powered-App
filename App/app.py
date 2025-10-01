import os
import math
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    # Initialize LLM
    model = ChatOpenAI(
        model_name="meta-llama/llama-3.3-70b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.8,
    )

    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

    # Create conversation chain
    st.session_state.conversation = ConversationChain(llm=model, memory=memory)

    # Initialize conversation with a prompt
    prompt = "You are a ChatGPT-like assistant that helps users get information and explains how things work."
    st.session_state.conversation.run(prompt)

# UI Header
st.title("🤖 AI Chatbot with Memory")
st.markdown("*Powered by Llama 3.3 70B*")
st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.run(user_input)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar with controls
with st.sidebar:
    st.header("💬 Chat Controls")

    st.markdown("### Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Chat History Limit")
    st.info(
        "⚡ This chatbot stores only the last **10 messages** to ensure smooth performance.\n\n"
        "Once the limit is reached, the history will be cleared automatically."
    )

    st.markdown("---")
    st.markdown("### 📊 Chat Stats")
    st.metric("Total Messages", math.ceil(len(st.session_state.messages) / 2))

    if math.ceil(len(st.session_state.messages) / 2) >= 10:
        st.warning("💡 Chat history limit reached. Clearing history...")
        st.session_state.messages = []
        st.session_state.conversation.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This chatbot remembers conversation context using **LangChain's ConversationBufferMemory**.\n\n"
        "It is powered by Llama 3.3 70B and uses the OpenRouter API for responses."
    )

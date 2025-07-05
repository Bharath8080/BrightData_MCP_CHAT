from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import os
import warnings
import re
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    # Make sure to update to the full absolute path to your math_server.py file
    args=["@brightdata/mcp"],
)

# Suppress specific schema warnings
warnings.filterwarnings("ignore", message="Key '\$schema' is not supported in schema, ignoring")
warnings.filterwarnings("ignore", message="Key 'additionalProperties' is not supported in schema, ignoring")

async def chat_with_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            # Start conversation history
            messages = [
                {
                    "role": "system",
                    "content": "You can use multiple tools in sequence to answer complex questions. Think step by step.",
                }
            ]

            print("Type 'exit' or 'quit' to end the chat.")
            while True:
                user_input = input("\nYou: ")
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                # Add user message to history
                messages.append({"role": "user", "content": user_input})

                # Call the agent with the full message history
                agent_response = await agent.ainvoke({"messages": messages})

                # Extract agent's reply and add to history
                ai_message = agent_response["messages"][-1].content
                print(f"Agent: {ai_message}")

async def run_agent(user_input):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            # Compose the message history for a single turn
            messages = [
                {
                    "role": "system",
                    "content": "You can use multiple tools in sequence to answer complex questions. Think step by step.",
                },
                {"role": "user", "content": user_input},
            ]
            agent_response = await agent.ainvoke({"messages": messages})
            ai_message = agent_response["messages"][-1].content
            return ai_message

def run_streamlit_chat():
    st.title("Brightdata MCP Agent")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history using st.chat_message
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    # Use st.chat_input for user input
    user_input = st.chat_input("Type your message and press Enter")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            response = asyncio.run(run_agent(user_input))
            st.session_state.chat_history.append({"role": "agent", "content": response})
        except Exception as exc:
            st.session_state.chat_history.append({"role": "agent", "content": f"[Error] {exc}"})
        st.rerun()

if __name__ == "__main__":
    run_streamlit_chat()

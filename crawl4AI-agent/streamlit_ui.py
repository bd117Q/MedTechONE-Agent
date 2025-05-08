from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
import httpx

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from MedTechONE_AI_Expert import MedTechONE_AI_Expert, MedTechONEAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize clients with error handling
try:
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables are not set")
        
    supabase: Client = Client(supabase_url, supabase_key)
    
    # Test Supabase connection
    test_result = supabase.from_('site_pages').select('count').limit(1).execute()
    if not test_result:
        raise ConnectionError("Failed to connect to Supabase database")

    # Verify Airtable credentials
    airtable_token = os.getenv("AIRTABLE_TOKEN")
    airtable_base_id = os.getenv("AIRTABLE_BASE_ID")
    
    if not airtable_token or not airtable_base_id:
        raise ValueError("AIRTABLE_TOKEN or AIRTABLE_BASE_ID environment variables are not set")
        
    # Test Airtable connection
    from pyairtable import Api
    api = Api(airtable_token)
    base = api.base(airtable_base_id)
    
    # List available tables
    tables = base.tables()
    print(f"Available Airtable tables: {[table.name for table in tables]}")
    
    if "Source repository" not in [table.name for table in tables]:
        raise ValueError("'Source repository' table not found in Airtable base")
        
    # Test table access
    from pyairtable import Table
    test_table = Table(airtable_token, airtable_base_id, "Source repository")
    test_records = test_table.all(limit=1)
    if not test_records:
        print("Warning: No records found in Airtable 'Source repository' table")
    else:
        print(f"Successfully connected to Airtable. Found {len(test_records)} test records.")
        
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = MedTechONEAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        airtable_token=os.getenv("AIRTABLE_TOKEN"),
        airtable_base_id=os.getenv("AIRTABLE_BASE_ID")
    )

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            # Run the agent in a stream
            async with MedTechONE_AI_Expert.run_stream(
                user_input,
                deps=deps,
                message_history=st.session_state.messages[:-1],  # pass entire conversation so far
            ) as result:
                # We'll gather partial text to show incrementally
                partial_text = ""
                message_placeholder = st.empty()

                # Render partial text as it arrives
                async for chunk in result.stream_text(delta=True):
                    partial_text += chunk
                    message_placeholder.markdown(partial_text)

                # Now that the stream is finished, we have a final result.
                # Add new messages from this run, excluding user-prompt messages
                filtered_messages = [msg for msg in result.new_messages() 
                                    if not (hasattr(msg, 'parts') and 
                                            any(part.part_kind == 'user-prompt' for part in msg.parts))]
                st.session_state.messages.extend(filtered_messages)

                # Add the final response to the messages
                st.session_state.messages.append(
                    ModelResponse(parts=[TextPart(content=partial_text)])
                )
                return  # Success - exit the retry loop

        except httpx.RemoteProtocolError as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection error occurred. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                st.error("Failed to get a complete response after multiple attempts. Please try again.")
                raise
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise


async def main():
    col1, col2 = st.columns([3, 1])  # Adjust column ratios as needed

    with col1:
        st.title("MedTechONE AI Agent")
        st.write("Ask any question about MedTech Resources.")

    with col2:
        st.image("https://raw.githubusercontent.com/bd117Q/MedTechONE-Agent/main/crawl4AI-agent/assets/hamlyn_icon.png", width=120)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about our MedTech Resources?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())

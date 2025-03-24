
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional
import requests
import os
import streamlit as st
import json

load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

# ---- Utils ---- #

def extract_cast_info(api_response: json) -> dict:
    """
    Extract cast information from the API response.
    """
    casts = []
    if api_response['status_code'] != 200:
        raise ValueError(f"API response failed: {api_response['body']}")
    else:
        for cast in api_response["body"]:
            cast_info = {
                "author": cast["metadata"]["author"]["username"],
                "text": cast["metadata"]["text"],
            }
            casts.append(cast_info)
    return casts

# ---- Tools ---- #

# Tool 1: Personalized Feed 
def get_personalized_feed(user_id: str) -> str:
    """
    Get personalized feed for a user given user_id.

    Args:
        user_id: user id
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/feed/for-you"
    payload = {
        "user_id": user_id,
        "top_k": 2,
        "feed_id": "feed_352",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    input_for_llm = extract_cast_info(json.loads(response.text))
    return input_for_llm

# Tool 2: Trending Cast
def get_trending_cast() -> List[dict]:
    """
    Get trending posts.
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/feed/trending"
    payload = {
        "top_k": 2,
        "feed_id": "feed_405",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    input_for_llm = extract_cast_info(json.loads(response.text))
    return input_for_llm

# Tool 3: Popular Cast
def get_popular_cast() -> List[dict]:
    """
    Get popular posts.
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/feed/popular"
    payload = {
        "top_k": 2,
        "feed_id": "feed_404",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    input_for_llm = extract_cast_info(json.loads(response.text))
    return input_for_llm

# Small model for summarization
summarizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# Define LLM with bind tools
mbd_tools = [get_personalized_feed, get_trending_cast, get_popular_cast]
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
llm_with_tools = llm.bind_tools(mbd_tools)

# Custom feed state
class FeedState(MessagesState):
    conversation: List[str]=[]
    summary : Optional[str]=None

# Node for summarizing history
def summarize_history(state: FeedState):
    messages = state["messages"]
    if len(messages) <= 2:
        return {"messages": messages, "summary": None}
    
    history_text = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in messages]
    )
    summary_prompt = ("You are a summarization assistant helping another AI agent stay informed with minimal context."
    "Summarize only the essential highlights of the conversation in a very concise manner, so that the agent knows all the steps taken."
    "The summary follow the format: 'User says: {text}'. AI says {text}."
    "About the AI responses, focus just on the main points, like the tool called and NOT the results (e.g. users and texts)."
    f"\n\nConversation:\n\n{history_text}"
    )
    summary_output = summarizer_llm.invoke(summary_prompt)
    summarized_context = summary_output.content
    return {"summary": summarized_context, "messages": messages}

# System message
sys_msg = SystemMessage("You are a helpful assistant for retrieving MBD feed data from the Farcaster network."
"Use the function get_personalized_feed(user_id) to fetch a user’s feed only when the user explicitly requests it."
"If they ask for a personalized feed but don’t provide a valid user_id (a number with max 8 digits), ask them for it."
"If the user is just greeting or making small talk, respond accordingly and ask how you can help."
"If you receive a result from tool, respond to user in natural language using this format: @{author} says: {text}."
"You can also use the following tools: get_trending_cast, get_popular_cast."
)

# Node for feed_builder
def feed_builder(state: FeedState):
    if state["summary"] is not None:
        if state["messages"][-1].type == 'tool':
            return {"messages": [llm_with_tools.invoke([sys_msg, SystemMessage(content=state["summary"])]+ state["messages"][-2:])]} 
        else: 
            return {"messages": [llm_with_tools.invoke([sys_msg, SystemMessage(content=state["summary"])]+ [state["messages"][-1]])]} 
    else:
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(FeedState)
builder.add_node("summarize_history", summarize_history)
builder.add_node("feed_builder", feed_builder)
builder.add_node("tools", ToolNode(mbd_tools))
builder.add_edge(START, "summarize_history")
builder.add_edge("summarize_history", "feed_builder")
builder.add_conditional_edges(
    "feed_builder",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "feed_builder")
# Memory
memory = MemorySaver()
# Compile graph
graph_memory = builder.compile(checkpointer=memory)
# Config for memory
config = {"configurable": {"thread_id": "2"}}


# Streamlit UI
st.title("Feed Builder")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask anything"):
    # Display user message in chat
    with st.chat_message("user"):
        st.write(prompt)

    # Add user message to session
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Build full message history for LangGraph
    full_history = []
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            full_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            full_history.append(AIMessage(content=msg["content"]))

    # Invoke LangGraph with memory
    response = graph_memory.invoke({"messages": full_history}, config)
    bot_reply = response["messages"][-1].content
    second_to_last_bot_reply = response["messages"][-2]

    # Display assistant reply
    with st.chat_message("assistant"):
        if second_to_last_bot_reply.type == "tool":
            st.write(response["messages"][-3].additional_kwargs)
        st.write(bot_reply)

    # Store assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})


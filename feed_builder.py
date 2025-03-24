
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import requests
import os
import json

load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")


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
        "top_k": 1,
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


# Define LLM with bind tools
#mbd_tools = [get_personalized_feed, get_trending_cast, get_popular_cast, get_semantic_cast]
mbd_tools = [get_personalized_feed]
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
llm_with_tools = llm.bind_tools(mbd_tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant for retrieving MBD feed data from the Farcaster network."
"Use the function get_personalized_feed(user_id) to fetch a user’s feed only when the user explicitly requests it."
"If they ask for a personalized feed but don’t provide a valid user_id (a number with max 8 digits), ask them for it."
"If the user is just greeting or making small talk, respond accordingly and ask how you can help."
"If the user ask for MBD informations, respond only with usernames and their messages, using this format: @username says: text."
)

# Node
def feed_builder(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("feed_builder", feed_builder)
builder.add_node("tools", ToolNode(mbd_tools))
builder.add_edge(START, "feed_builder")
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
config = {"configurable": {"thread_id": "1"}}


# ---- Hello World Example ---- #   

#messages = [HumanMessage(content="Hello World! I'm Matteo.")]
#messages = graph_memory.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()

# ---- Personalized Feed Example ---- #

messages = [HumanMessage(content="Give me the personalized feed.")]
messages = graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
messages = [HumanMessage(content="My user id is 123.")]
messages = graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()

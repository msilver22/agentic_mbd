from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional
from typing_extensions import TypedDict
import requests
import os
import json
from langfuse.langchain import CallbackHandler
import logging
import re

# ---- API keys ---- #
load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

# ---- Langfuse cloud ----#
def get_langfuse_handler():
    return CallbackHandler()

langfuse_handler = get_langfuse_handler()

# ---- State of the graph ---- #
class IcebreakerState(MessagesState):
    all_keywords: json



# ---- LLMs ---- #
summarizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
conversational_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.2)
json_filler_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)


# ---- Nodes ---- #
#TODO

# ---- Graph ---- #
builder = StateGraph(IcebreakerState)
#TODO


# Compile graph
agent_icebreaker = builder.compile()
# Visualize graph
visualize_graph = True
if visualize_graph:
    image = Image(agent_icebreaker.get_graph().draw_mermaid_png())
    with open("../graphs/icebreaker.png", "wb") as f:
        f.write(image.data)
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [langfuse_handler],
}

messages = agent_icebreaker.invoke({ "all_keywords":{}, "messages": []}, config)
for m in messages['messages']:
    m.pretty_print()

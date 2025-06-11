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

# ---- API keys ---- #
load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

# ---- Langfuse cloud ----#
langfuse_handler = CallbackHandler()

# ---- State of the graph ---- #
class PrompterState(MessagesState):
    fid : int
    casts: List[str]
    keywords: List[str]

# ---- LLMs ---- #
extrapolator_LLM = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
ranker_LLM = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

# ---- Nodes ---- #

def cast_fetcher(state: PrompterState) -> PrompterState:
    return state

def topic_extrapolator(state: PrompterState) -> PrompterState:
    return state

def topic_ranker(state: PrompterState) -> PrompterState:
    return state

def social_prompter(state: PrompterState) -> PrompterState:
    return state

# ---- Graph ---- #
builder = StateGraph(PrompterState)
builder.add_node("fetcher", cast_fetcher)
builder.add_node("extrapolator", topic_extrapolator)
builder.add_node("ranker", topic_ranker)
builder.add_node("prompter", social_prompter)
builder.add_edge(START, "fetcher")
builder.add_edge("fetcher", "extrapolator")
builder.add_edge("extrapolator", "prompter")
builder.add_edge("prompter", END)

# Compile graph
agent_social_prompter = builder.compile()
# Visualize graph
visualize_graph = True
if visualize_graph:
    image = Image(agent_social_prompter.get_graph().draw_mermaid_png())  
    with open("graphs/social_prompter.png", "wb") as f:
        f.write(image.data)
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [langfuse_handler],
}

messages = agent_social_prompter.invoke({"fid": 4461, "casts": [], "keywords": [], "messages": []}, config)
for m in messages['messages']:
    m.pretty_print()

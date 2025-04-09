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
from langfuse.callback import CallbackHandler
import logging

# ---- API keys ---- #
load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

# ---- Langfuse cloud ----#
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com" 
)


# ---- Utils ---- #
def extract_casts(api_response: dict) -> str:
    """
    Extract cast information from the API response and return as Markdown text.
    """
    if api_response['status_code'] != 200:
        raise ValueError(f"API response failed: {api_response['body']}")
    
    markdown_output = ""
    casts = api_response["body"]
    
    for i, cast in enumerate(casts, 1):
        author = cast["metadata"]["author"]["username"]
        text = cast["metadata"]["text"]
        
        markdown_output += f"# {i}. Author: {author}\n> {text}\n\n"

    return markdown_output


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
        "top_k": 3,
        "feed_id": "feed_352",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_casts(json.loads(response.text))
    return fetched_results

# Tool 2: Trending Cast
def get_trending_cast() -> str:
    """
    Get trending posts.
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/feed/trending"
    payload = {
        "top_k": 3,
        "feed_id": "feed_405",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_casts(json.loads(response.text))
    return fetched_results

# Tool 3: Popular Cast
def get_popular_cast() -> str:
    """
    Get popular posts.
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/feed/popular"
    payload = {
        "top_k": 3,
        "feed_id": "feed_404",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_casts(json.loads(response.text))
    return fetched_results


# ---- Models ---- #

# Small model for summarization
summarizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Big model with tools
mbd_tools = [get_personalized_feed, get_trending_cast, get_popular_cast]
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
llm_with_tools = llm.bind_tools(mbd_tools)


# ---- State of the graph ---- #

class FeedState(MessagesState):
    summary : Optional[str] = None

# ---- Nodes ---- #

# Node for summarizing history
def summarize_history(state: FeedState):
    messages = state["messages"]
    # Check if the last message is a feed illustration
    # If so, change it from the messages list so that the input tokens are optimized
    if len(messages) > 1:
        if messages[-2].content.startswith("We fetched"):
            state["messages"][-2].content = "AI showed fetched results."
    history_text = ""
    if len(messages) == 1:
        history_text = messages[0].content
    else:
        history_text = "\n\n".join(
            [f"{m.type.upper()}: {m.content}" for m in messages[:-1]]
        )

    summary_prompt = (
    "You are a summarization assistant. Your only task is to condense a conversation as much as possible."
    "Paraphrase the conversation in a concise way, preserving the meaning and intent of the original messages."
    "Preserve all user-provided details, especially personal information like user IDs, names, or preferences."
    "For AI responses, omit them entirely unless they involve a critical step (e.g., a tool call). "
    "If the AI response includes tool results like user-generated content (e.g., posts, authors, texts), "
    "do not include them in the summary, only mention that a tool was called."
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
"If the user asks for its personal information, first check if it's present in the summary."
"If it's not included, politely inform the user that it cannot be remembered."
)

# Node for feed_builder
def feed_builder(state: FeedState):
    if state["summary"] is not None:
        sys_summary = SystemMessage(
            content=(
                "This is your memory summary of the conversation so far.\n"
                "Before answering, always check if the required information is already included below.\n\n"
                f"--- BEGIN SUMMARY ---\n{state['summary']}\n--- END SUMMARY ---"
            )
        )
        if state["messages"][-1].type == 'tool':
            return {"messages": [llm_with_tools.invoke([sys_msg]+ state["messages"][-2:])]} 
        else: 
            return {"messages": [llm_with_tools.invoke([sys_msg, sys_summary]+ [state["messages"][-1]])]} 
    else:
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
    

# Node for printing results
def feed_printer(state: FeedState):
    """
    Print the fetched results.
    """
    results = state["messages"][-1].content
    state["messages"][-1].content = "Tool executed successfully."
    return {"messages": [AIMessage(content="We fetched the following casts:\n\n" + results)]}
    


# ---- Graph ---- #

builder = StateGraph(FeedState)
builder.add_node("summarizer", summarize_history)
builder.add_node("feed_builder", feed_builder)
builder.add_node("feed_printer", feed_printer)
builder.add_node("tools", ToolNode(tools = mbd_tools))
builder.add_edge(START, "summarizer")
builder.add_edge("summarizer", "feed_builder")
builder.add_conditional_edges(
    "feed_builder",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "feed_printer")
builder.add_edge("feed_printer", END)

# Memory
memory = MemorySaver()
# Compile graph
agent_feed_builder = builder.compile(checkpointer=memory)
# Visualize graph
visualize_graph = True
if visualize_graph:
    image = Image(agent_feed_builder.get_graph().draw_mermaid_png())  
    with open("graphs/feed_builder.png", "wb") as f:
        f.write(image.data)
# Config for memory
config = {
    "configurable": {"thread_id": "2"},
    "callbacks": [langfuse_handler],
}



# ---- Examples ---- #

# Hello World Example  

#messages = [HumanMessage(content="Hello World! I'm Matteo.")]
#messages = agent_feed_builder.invoke({"messages": messages, "summary": None}, config)
#for m in messages['messages']:
#    m.pretty_print()
#
## Personalized Feed Example 
#
#messages = [HumanMessage(content="Give me the personalized feed.")]
#messages = agent_feed_builder.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="My user id is 123.")]
#messages = agent_feed_builder.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="What is my user id?")]
#messages = agent_feed_builder.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()

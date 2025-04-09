from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional, Literal
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
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY3"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY3"),
    host="https://cloud.langfuse.com" 
)

# ---- Utils ---- #

def extract_casts(api_response: dict) -> str:
    """
    Extract cast information from the API response and return as Markdown text.
    """
    if api_response['status_code'] != 200:
        return f"API response failed: {api_response['body']}"
    
    markdown_output = ""
    casts = api_response["body"]

    if not casts:
        return "I'm sorry, we didn't find casts about it."
    
    for i, cast in enumerate(casts, 1):
        author = cast["metadata"]["author"]["username"]
        text = cast["metadata"]["text"]
        
        markdown_output += f"### {i}. @{author} says:\n> {text}\n\n"

    return markdown_output

def extract_users(api_response: dict) -> str:
    """
    Extract cast information from the API response and return as Markdown text.
    """
    if api_response['status_code'] != 200:
        return f"API response failed: {api_response['body']}"
    
    markdown_output = ""
    users = api_response["body"]

    if not users:
        return "I'm sorry, we didn't find users about it."
    
    
    for i, user in enumerate(users, 1):
        user_id = user["user_id"]        
        markdown_output += f"#### {i}. FID {user_id}\n\n"

    return markdown_output


# ---- Tools ---- #

# ---- Feed Builder tools ---- #

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
        "return_metadata": True,
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
        "return_metadata": True,
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
        "return_metadata": True,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_casts(json.loads(response.text))
    return fetched_results

# Tool 4: Semantic casts
def get_semantic_cast(query:str) -> str:
    """
    Get semantic posts.

    Args:
        query: query string
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/search/semantic"
    payload = {
        "query": query,
        "top_k": 3,
        "return_metadata": True,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_casts(json.loads(response.text))
    return fetched_results



# ---- Social Prompter tools ---- #

# Tool 1: Similar users
def get_similar_user(user_id: str) -> str:
    """
    Get a list of similar users for a given user_id.

    Args:
        user_id: user id
    """
    url = "https://api.mbd.xyz/v2/farcaster/users/feed/similar"
    payload = {
        "user_id": user_id,
        "top_k": 5,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_users(json.loads(response.text))
    return fetched_results

# Tool 2: Similar users
def get_semantic_user(query: str) -> str:
    """
    Get a list of similar users for a given query.

    Args:
        query: user's query
    """
    url = "https://api.mbd.xyz/v2/farcaster/users/search/semantic"
    payload = {
        "query": query,
        "top_k": 5,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_users(json.loads(response.text))
    return fetched_results

# Tool 3: Suggested users
def get_suggested_user(user_id: str) -> str:
    """
    Get a list of suggested users for a given user_id.

    Args:
        user_id: user id
    """
    # Step 1: Get the label with highest score for the user

    url1 = "https://api.mbd.xyz/v2/farcaster/users/labels/for-users"
    payload1 = {
        "users_list": [user_id],
        "label_category": "topics"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response1 = requests.post(url1, json=payload1, headers=headers)
    result1 = json.loads(response1.text)
    labels = result1["body"][0]["ai_labels"]["topics"]
    opt_label = ""
    best_score = 0
    for item in labels:
        label = item["label"]
        score = item["score"]
        if score > best_score:
            best_score = score
            opt_label = label
    
    # Step 2: Get the top_k users with highest score for the label

    url2 = "https://api.mbd.xyz/v2/farcaster/users/labels/top-users"
    payload2 = {
        "label": opt_label,
        "top_k": 5,
        "minimum_activity_count": 100
    }
    response2 = requests.post(url2, json=payload2, headers=headers)
    fetched_results = extract_users(json.loads(response2.text))
    return fetched_results

# ---- Models ---- #

# Small model for summarization
summarizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Planner model
planner_llm = ChatGroq(model="deepseek-r1-distill-qwen-32b", temperature=0)

# Small talks model
small_talk_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Feed Builder model
feed_tools = [get_personalized_feed, get_trending_cast, get_popular_cast, get_semantic_cast]
feed_builder_llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
feed_builder_llm_with_tools = feed_builder_llm.bind_tools(feed_tools)

# Social Prompter model
prompting_tools = [get_similar_user, get_semantic_user, get_suggested_user]
social_prompter_llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, )
social_prompter_llm_with_tools = social_prompter_llm.bind_tools(prompting_tools)


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

    summary_prompt = ("You are a summarization assistant. Your only task is to condense a conversation as much as possible."
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

planner_sys_msg = (
    "Choose one node from: FEED, PROMPTING, or OTHER, based on the user's message. "
    "Output ONLY the node label (e.g., FEED). "
    "FEED: For suggesting feeds or casts.\n"
    "PROMPTING: For suggesting users.\n"
    "OTHER: For small talk, unrelated questions, or anything else."
)

def planner_node(state: FeedState):
    if state["summary"] is not None:
        sys_summary = (
                "This is your memory summary of the conversation so far.\n"
                "Before answering, always check if the required information is already included below.\n\n"
                f"--- BEGIN SUMMARY ---\n{state['summary']}\n--- END SUMMARY ---"
        )
        return {"messages": planner_llm.invoke([planner_sys_msg, sys_summary] + [state["messages"][-1]])} 
    else:
        return {"messages": planner_llm.invoke([planner_sys_msg] + [state["messages"][-1]])}

def planner_condition(state: FeedState) -> Literal["feed_builder", "social_prompter", "small_talks"]:
    # Retrieve the decision returned by the planner node
    decision = state["messages"][-1].content.strip().splitlines()[-1].upper()
    state["messages"][-1].content = f"AI chose the node {decision}."
    
    if decision == "FEED":
        return "feed_builder"
    elif decision == "PROMPTING":
        return "social_prompter"
    else:
        return "small_talks"

# Node for small talks
def small_talks_node(state: FeedState):
    small_talks_sys = ("You are a helpful assistant for small talks."
    "You are a part of an agent that is able to "
    "- build feed on Farcaster social network : semantic search, personalized feed, trending posts, etc.)"
    "- social prompting: suggest similar, semantic users, or suggested users)."
    "If the user is just greeting or making small talk, respond accordingly and inform on agent's capabilities."
    f"\n\nUser query:\n\n{state['messages'][-2].content}"
    )
    return {"messages": [small_talk_llm.invoke(small_talks_sys)]}


# System messages of feed builder and social prompter

feed_sys_msg = SystemMessage("You are a helpful assistant for retrieving MBD feed data from the Farcaster network."
"You can use the following tools: get_trending_cast, get_popular_cast, get_personalized_feed(user_id), get_semantic_cast(query)."
"If the user is asking what are your capabilities, or asking for how you can help him, respond explaining your tools."
"If they ask for a personalized feed but don’t provide a user_id, check if it's present in the summary."
"The user_id must be a number. If it's not present in the summary,  DO NOT invent it and ASK the user for it."
"If the user asks for its personal information, first check if it's present in the summary."
)

prompting_sys_msg = SystemMessage("You are a helpful assistant for retrieving users from the Farcaster network."
"You can use the following tools: get_similar_user(user_id), get_semantic_user(query), get_suggested_user(user_id)."
"If the user is asking what are your capabilities, or asking for how you can help him, respond explaining your tools."
"If they ask for similar or suggested users but don’t provide a user_id, check if it's present in the summary."
"The user_id must be a number. If it's not present in the summary, DO NOT invent it and ASK the user for it."
"If the user asks for its personal information, first check if it's present in the summary."
)

# Node for feed_builder
def feed_builder_node(state: FeedState):
    if state["summary"] is not None:
        sys_summary = SystemMessage(
            content=(
                "This is your memory summary of the conversation so far.\n"
                "Before answering, always check if the required information is already included below.\n\n"
                f"--- BEGIN SUMMARY ---\n{state['summary']}\n--- END SUMMARY ---"
            )
        )
        if state["messages"][-1].type == 'tool':
            return {"messages": [feed_builder_llm_with_tools.invoke([feed_sys_msg]+ state["messages"][-2:])]} 
        else: 
            return {"messages": [feed_builder_llm_with_tools.invoke([feed_sys_msg, sys_summary]+ [state["messages"][-2]])]} 
    else:
        return {"messages": [feed_builder_llm_with_tools.invoke([feed_sys_msg] + state["messages"])]}
    
def feed_builder_tools_condition(state: FeedState) -> Literal["feed_tools", "__end__"]:
    try :
        tool_call = state["messages"][-1].additional_kwargs["tool_calls"]
    except KeyError:
        tool_call = None
   
    if tool_call is not None:
        return "feed_tools"
    else:
        return "__end__"

def additional_prompting(state: FeedState):
    """
    Fetch additional informations based on previous tool calls.
    """
    tool_call = state["messages"][-2].tool_calls[0]
    tool_name = tool_call["name"]

    if tool_name == "get_semantic_cast":
        query = tool_call["args"]["query"]
        return {"messages": [AIMessage(content=get_semantic_user(query))]}

    elif tool_name == "get_personalized_feed":
        user_id = tool_call["args"]["user_id"]
        return {"messages": [AIMessage(content=get_suggested_user(user_id))]}
    
    elif tool_name == "get_popular_cast" or tool_name == "get_trending_cast":
        return {"messages": [AIMessage(content=get_semantic_user("Farcaster"))]}

    else: 
        return {"messages": [AIMessage(content="No additional information fetched.")]}


# Node for social prompter
def social_prompter_node(state: FeedState):
    if state["summary"] is not None:
        sys_summary = SystemMessage(
            content=(
                "This is your memory summary of the conversation so far.\n"
                "Before answering, always check if the required information is already included below.\n\n"
                f"--- BEGIN SUMMARY ---\n{state['summary']}\n--- END SUMMARY ---"
            )
        )
        if state["messages"][-1].type == 'tool':
            return {"messages": [social_prompter_llm_with_tools.invoke([prompting_sys_msg]+ state["messages"][-2:])]} 
        else: 
            return {"messages": [social_prompter_llm_with_tools.invoke([prompting_sys_msg, sys_summary]+ [state["messages"][-2]])]} 
    else:
        return {"messages": [social_prompter_llm_with_tools.invoke([prompting_sys_msg] + state["messages"])]}

def social_prompter_tools_condition(state: FeedState) -> Literal["prompting_tools", "__end__"]:
    try:
        tool_call = state["messages"][-1].additional_kwargs["tool_calls"]
    except (KeyError, IndexError):
        tool_call = None

    if tool_call is not None:
        return "prompting_tools"
    else:
        return "__end__"
    

def additional_building(state: FeedState):
    """
    Fetch additional informations based on previous tool calls.
    """
    tool_call = state["messages"][-2].tool_calls[0]
    tool_name = tool_call["name"]

    if tool_name == "get_semantic_user":
        query = tool_call["args"]["query"]
        return {"messages": [AIMessage(content=get_semantic_cast(query))]}

    elif tool_name == "get_similar_user" or tool_name == "get_suggested_user":
        user_id = tool_call["args"]["user_id"]
        return {"messages": [AIMessage(content=get_personalized_feed(user_id))]}
    
    else: 
        return {"messages": [AIMessage(content="No additional information fetched.")]}

# Node for printing results of feed builder
def feed_printer_node(state: FeedState):
    """
    Print the fetched results.
    """
    casts_results = state["messages"][-2].content
    users_results = state["messages"][-1].content
    state["messages"][-1].content = "Tool executed successfully."
    state["messages"][-2].content = "Tool executed successfully."
    return {"messages": [AIMessage(content=f"We fetched the following casts:\n\n{casts_results}\nIn addition, we suggest the following users:\n\n{users_results}")]}
    
# Node for printing results of social prompter
def social_tips_printer_node(state: FeedState):
    """
    Print the fetched results.
    """
    users_results = state["messages"][-2].content
    casts_results = state["messages"][-1].content
    state["messages"][-2].content = "Tool executed successfully."
    state["messages"][-1].content = "Tool executed successfully."
    return {"messages": [AIMessage(content=f"We fetched the following users:\n\n{users_results}\n"
                                            f"In addition, we suggest the following casts:\n\n{casts_results}")]}
    

# ---- Graph ---- #
# Nodes
builder = StateGraph(FeedState)
builder.add_node("summarizer", summarize_history)
builder.add_node("planner", planner_node)
builder.add_node("small_talks", small_talks_node)
builder.add_node("social_prompter", social_prompter_node)
builder.add_node("feed_builder", feed_builder_node)
builder.add_node("social_tips_printer", social_tips_printer_node)
builder.add_node("feed_printer", feed_printer_node)
builder.add_node("feed_tools", ToolNode(tools = feed_tools, name="feed_tools"))
builder.add_node("prompting_tools", ToolNode(tools = prompting_tools, name="prompting_tools"))
builder.add_node("additional_prompting", additional_prompting)
builder.add_node("additional_building", additional_building)
# Edges
builder.add_edge(START, "summarizer")
builder.add_edge("summarizer", "planner")
builder.add_conditional_edges(
    "planner", 
    planner_condition
)
builder.add_conditional_edges(
    "feed_builder",
    feed_builder_tools_condition,
)
builder.add_conditional_edges(
    "social_prompter",
    social_prompter_tools_condition,
)
builder.add_edge("prompting_tools", "additional_building")
builder.add_edge("additional_building", "social_tips_printer")
builder.add_edge("social_tips_printer", END)
builder.add_edge("feed_tools", "additional_prompting")
builder.add_edge("additional_prompting", "feed_printer")
builder.add_edge("feed_printer", END)
builder.add_edge("small_talks", END)

# Memory
memory = MemorySaver()
# Compile graph
mbd_agent = builder.compile(checkpointer=memory)
# Visualize graph
visualize_graph = True
if visualize_graph:
    image = Image(mbd_agent.get_graph().draw_mermaid_png())  
    with open("graphs/mbd_agent_v2.png", "wb") as f:
        f.write(image.data)
# Config for memory
config = {
    "configurable": {"thread_id": "3"},
    "callbacks": [langfuse_handler],
}


# ---- Examples ---- #

# Hello World Example  
#messages = [HumanMessage(content="Hello World! I'm Matteo.")]
#messages = mbd_agent.invoke({"messages": messages, "summary": None}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="What can you do for me?")]
#messages = mbd_agent.invoke({"messages": messages, "summary": None}, config)
#for m in messages['messages']:
#    m.pretty_print()

# ---- Social Prompting examples ---- #

# Get similar users
#messages = [HumanMessage(content="Show me users similar to me.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="My user id is 123.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()

# Get suggested users
#messages = [HumanMessage(content="Show me users suggested for me (user id 123).")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="My user id is 123.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()

# Get semantic users
#messages = [HumanMessage(content="Show me users related to web3 app development.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()



# ---- Feed Builder examples ---- #

# Personalized Feed Example 
#
#messages = [HumanMessage(content="Give me the personalized feed for user 123.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="My user id is 123.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
#messages = [HumanMessage(content="What is my user id?")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()

# Get trending cast
#messages = [HumanMessage(content="Show me the trending cast.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()


#messages = [HumanMessage(content="Hi! I'm Matteo.")]
#messages = mbd_agent.invoke({"messages": messages}, config)
#for m in messages['messages']:
#    m.pretty_print()
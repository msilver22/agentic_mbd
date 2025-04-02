from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional
import requests
import os
import streamlit as st
import json
from langfuse.callback import CallbackHandler


st.title("Social Prompter")


# ---- API keys ---- #
load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

# ---- Langfuse cloud ----#
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY2"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY2"),
    host="https://cloud.langfuse.com" 
)


# ---- Utils ---- #
def extract_users(api_response: dict) -> str:
    """
    Extract cast information from the API response and return as Markdown text.
    """
    if api_response['status_code'] != 200:
        raise ValueError(f"API response failed: {api_response['body']}")
    
    markdown_output = ""
    users = api_response["body"]
    
    for i, user in enumerate(users, 1):
        user_id = user["user_id"]        
        markdown_output += f"### {i}. FID {user_id}\n\n"

    return markdown_output

# ---- Tools ---- #

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

# Big model with tools
mbd_tools = [get_similar_user, get_semantic_user, get_suggested_user]
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
llm_with_tools = llm.bind_tools(mbd_tools)

# ---- State ---- #

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
sys_msg = SystemMessage("You are a helpful assistant for retrieving users from the Farcaster network."
"You can use the following tools: get_similar_user(user_id), get_semantic_user(query), get_suggested_user(user_id)."
"If they ask for similar or suggested users but don’t provide a user_id, first check if it's present in the summary."
"If it's not present, ask the user for their user_id."
"If the user is just greeting or making small talk, respond accordingly and ask how you can help."
"If the user asks for its personal information, first check if it's present in the summary."
"If it's not included, politely inform the user that it cannot be remembered."

)

## Node for social prompter
def social_prompter(state: FeedState):
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
def social_tips_printer(state: FeedState):
    """
    Print the fetched results.
    """
    results = state["messages"][-1].content
    state["messages"][-1].content = "Tool executed successfully."
    return {"messages": [AIMessage(content="We fetched the following users:\n\n" + results)]}
    


# ---- Graph ---- #

# Nodes
builder = StateGraph(FeedState)
builder.add_node("summarizer", summarize_history)
builder.add_node("social_prompter", social_prompter)
builder.add_node("social_tips_printer", social_tips_printer)
builder.add_node("tools", ToolNode(tools = mbd_tools))
# Edges
builder.add_edge(START, "summarizer")
builder.add_edge("summarizer", "social_prompter")
builder.add_conditional_edges(
    "social_prompter",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "social_tips_printer")
builder.add_edge("social_tips_printer", END)

# Memory
memory = MemorySaver()
# Compile graph
agent_social_prompter = builder.compile(checkpointer=memory)

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "agent" not in st.session_state:
    st.session_state.agent_social_prompter = builder.compile(checkpointer=st.session_state.memory)

if "config" not in st.session_state:
    st.session_state.config = {
        "configurable": {"thread_id": "3"},
        "callbacks": [langfuse_handler],
    }

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ---- Streamlit UI ---- #

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
    messages = [HumanMessage(content=prompt)]

    # Invoke the agent with the user input
    agent = st.session_state.agent_social_prompter
    config = st.session_state.config
    if len(st.session_state["messages"]) == 1:
        response = agent.invoke({"messages": messages, "summary": None}, config)
    else: 
        response = agent.invoke({"messages": messages}, config)
   
    # Display assistant reply
    bot_reply = response["messages"][-1].content
    second_to_last_reply = response["messages"][-2]

    with st.chat_message("assistant"):
        if second_to_last_reply.type == "tool":
            st.write(response["messages"][-3].additional_kwargs)
        st.write(bot_reply)

    # Store assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})


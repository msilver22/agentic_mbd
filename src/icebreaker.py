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
    summary: Optional[str] = None
    conversation_count: Optional[int] = None
    all_keywords: json
    

# ---- LLMs ---- #
summarizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
conversational_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.2)
json_filler_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)


# ---- Nodes ---- #

def summarize_history(state: IcebreakerState):
    messages = state["messages"]
    history_text = ""
    if len(messages) == 1:
        history_text = messages[0].content
    else:
        history_text = "\n".join(
            [f"{m.type.upper()}: {m.content}" for m in messages]
        )

    summary_prompt = (
        f"""You are an expert conversation summarizer. Your task is to produce a concise, accurate summary of the conversation below.
            - Paraphrase and condense the exchange, capturing all key details, facts, and user preferences.
            - Retain any personal information, names, interests, or specific topics mentioned by the user.
            - Summarize AI responses very briefly, focusing on their intent.
            - Do not add any commentary or interpretation.
            - Output only the summary, without any extra text.
            Conversation:
            {history_text}
        """
    )

    summary_output = summarizer_llm.invoke(summary_prompt)
    summarized_context = summary_output.content

    # Store the summary in the state
    if state["summary"] is None:
        state["summary"] = summarized_context
    else:
        state["summary"] += "\n" + summarized_context
    return state

def json_filler(state: IcebreakerState):
    all_keywords = state["all_keywords"]
    messages = state["messages"]
    # Get the latest user message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"all_keywords": all_keywords, "messages": messages}

    # Prepare prompt for the LLM
    json_prompt = (
        f"""You are an expert at extracting structured information from user conversations.\n
        Below is a summary of the conversation so far:\n
        Summary:\n{state['summary']}\n
        Your task:\n
        - Carefully analyze the summary for any new relevant keywords, topics, or interests that fit as macro or micro categories.\n
        - If you identify new relevant keywords, add them to the appropriate place in the JSON structure. Do not remove or modify existing entries unless necessary for accuracy.\n
        - If no new keywords are found, return the JSON unchanged.\n
        Return ONLY a valid JSON object. Do not include any explanations, comments, or extra text.\n
        Example:
        - Summary: "The user enjoys hiking and photography, particularly in nature, and also reads science fiction books."
        - Output: {{"Hobbies": ["Hiking", "Photography"], "Books": ["Science Fiction"]}}
    """)

    response = json_filler_llm.invoke(json_prompt)
    # Check if the response is valid JSON
    try:
        # Rimuove ``` e spazi
        cleaned = re.sub(r"^```(?:json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        updated_keywords = json.loads(cleaned)
    except Exception as e:
        logging.warning(f"Failed to parse JSON from LLM: {response.content}")
        updated_keywords = all_keywords


    # Store the updated keywords in the state
    state["all_keywords"] = updated_keywords
    return state

def engage_conversation(state: IcebreakerState):
    messages = state["messages"]


    # Prepare the system message with the user message
    conversation_prompt = (
        f"""You are an expert question-asking assistant. Your job is to ask friendly, open-ended questions that help the user share more about their interests.

            User message:
            {messages[-1].content}

            Instructions:
            - Ask questions related to ALL the user's mentioned interests.
            - Always ask for other interests or topics they might want to discuss, even if not related to the current conversation.
            - Output only the assistant's next message — a direct question.

            Example:
            - User's previous message: "I'm interested in Web3 technology and how it can change the way we interact online. I also love hiking."
            - Assistant: "Is there something specific about Web3 or hiking that you find most exciting? Are there other completely different things you're into?"
        """
    )
    
    # Increment conversation count
    if "conversation_count" not in state:
        state["conversation_count"] = 0
    state["conversation_count"] += 1
    print(f"Conversation count: {state['conversation_count']}")

    response = conversational_llm.invoke(conversation_prompt)
    response_content = response.content
    # Extract the text after the second <think> tag, if present
    last_think_end_index = response_content.rfind('</think>')

    if last_think_end_index == -1:
        # If no </think> tag is found, return the entire text
        content_after_think = response_content
    else:
        # Get all content after the last </think> tag
        content_after_think = response_content[last_think_end_index + len('</think>'):]

    messages.append(AIMessage(content=content_after_think.strip())) 
    return state

# ---- Graph ---- #
builder = StateGraph(IcebreakerState)
# Add nodes to the graph
builder.add_node("summarizer", summarize_history)
builder.add_node("json_filler", json_filler)
builder.add_node("engage_conversation", engage_conversation)
# Add edges to the graph
builder.add_edge(START, "summarizer")
builder.add_edge("summarizer", "json_filler")
builder.add_edge("json_filler", "engage_conversation")
builder.add_edge("engage_conversation", END)


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


# ---- Example usage ---- #
#
#first_message = AIMessage(content="Hello! I am your icebreaker assistant. I'm here to get to know you better. What are you most interested in or passionate about?")
#user_message = HumanMessage(content="I love hiking and photography, especially in nature. I also enjoy reading science fiction books.")
#
#messages = agent_icebreaker.invoke({ "all_keywords":{}, "messages": [first_message, user_message], "summary": None}, config)
#for m in messages['messages']:
#    m.pretty_print()
#
#second_message = HumanMessage(content="I also like cooking Italian food, especially pasta dishes. I love basketball")
#messages = agent_icebreaker.invoke({ "all_keywords": {}, "messages": [second_message], "summary": messages["summary"], "conversation_count": messages["conversation_count"]}, config)
#for m in messages['messages']:
#    m.pretty_print()
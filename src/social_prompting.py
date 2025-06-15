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
langfuse_handler = CallbackHandler()

# ---- Utils ---- #
def extract_embed_casts(api_response: dict) -> str:
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
        
        markdown_output += f"{i}. @{author} says:\n> {text}\n\n"

    return markdown_output


# ---- Tools ---- #
def get_user_casts(user_id: str) -> List[str]:
    """
    Get casts for a user given user_id.

    Args:
        user_id: user id
    Returns:
        List of formatted cast texts, each containing [TARGET POST] and optionally [RELATED POST]
    """
    url = "https://api.neynar.com/v2/farcaster/feed/user/casts/"
    querystring = {"limit": "5", "include_replies": "false", "fid": user_id}
    headers = {
        "x-api-key": os.getenv("NEYNAR_API_KEY"),
        "x-neynar-experimental": "false"
    }
    
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = response.json()
    casts = response_json.get("casts", [])
    
    if not casts:
        return ["No casts found."]
    
    formatted_casts = []
    for cast in casts:
        formatted_cast = ""
        text = re.sub(r'\n+', '. ', cast.get("text", "No text provided"))
        formatted_cast += "\n[TARGET POST]\n"
        formatted_cast += text 
        
        embeds = cast.get("embeds", [])
        if embeds:
            embed = embeds[0]
            if isinstance(embed, dict) and "cast" in embed:
                embed_text = re.sub(r'\n+', '. ', embed["cast"].get("text", "No embed text provided"))
                formatted_cast += "\n[RELATED POST]\n"
                formatted_cast += embed_text
        
        formatted_casts.append(formatted_cast)
    
    return formatted_casts

def get_semantic_casts(query:str) -> str:
    """
    Get semantic posts.

    Args:
        query: query string
    """
    url = "https://api.mbd.xyz/v2/farcaster/casts/search/semantic"
    payload = {
        "query": query,
        "top_k": 2,
        "return_metadata": True,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_embed_casts(json.loads(response.text))
    return fetched_results



# ---- State of the graph ---- #
class PrompterState(MessagesState):
    fid : int
    goal: str
    casts: List[str]
    keywords: List[str]



# ---- LLMs ---- #
extrapolator_LLM = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.05)
ranker_LLM = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.05)
writing_tips_LLM = ChatGroq(model="qwen/qwen3-32b", temperature=0.05)



# ---- Nodes ---- #
def cast_fetcher(state: PrompterState) -> PrompterState:
    """
    Fetch casts for the given user ID.
    """
    state["casts"] = get_user_casts(str(state["fid"])) 
    return state

def topic_extrapolator(state: PrompterState) -> PrompterState:
    """
    Extract keywords from each cast using the LLM.
    The LLM will analyze both the target post and related post (if present)
    to generate 1-2 relevant keywords.
    """
    keywords = []
    
    for cast in state["casts"]:
        extrapolator_prompt = f"""Analyze the following social media post and extract UP TO 3 key topics or themes.
            Focus on topics that:
                - Are specific and meaningful
                - Are hashtags marked with "/" 
                - Could spark interesting discussions
                - Have potential for follow-up questions
            Return ONLY the keywords, separated by commas, nothing else.
            Important: If you find hashtags marked with "/", return them WITHOUT the "/" symbol.
            For example, if you find "/AI", return "AI".
            
            Post content:
            {cast}"""

        response = extrapolator_LLM.invoke(extrapolator_prompt)
        
        # Extract keywords from response, clean them and convert to lowercase
        extracted_keywords = [k.strip().lower() for k in response.content.split(',')]
        keywords.extend(extracted_keywords)
    
    state["keywords"] = keywords
    return state

def topic_compressor(state: PrompterState) -> PrompterState:
    """
    Compress the list of keywords into a maximum of 5 topics that represent the core themes.
    """
    if not state["keywords"]:
        return state

    # Format keywords in triplets
    keywords_text = "[KEYWORDS]\n"
    for i in range(0, len(state["keywords"]), 3):
        triplet = state["keywords"][i:i+3]
        keywords_text += f"{', '.join(triplet)}\n"
    
    compressor_prompt = f"""You are a conversation topic selector. Your task is to analyze a list of keywords and select up to 5 topics that would be most interesting and engaging for a conversation.
        Focus on topics that:
            - Are more frequent than other keywords
            - Could spark interesting discussions
            - Have potential for follow-up questions
        
        Important: The keywords come in triplets (3 keywords per post). Consider this structure when selecting topics.
        You can:
            - Select the most interesting keyword from a triplet
            - Combine related keywords of the same triplet (e.g. if the triplet is ["AI", "technology", "innovation"], the topic could be "AI and technology")
            - Select completely new topics that emerge from the analysis
        
        Return only the selected topics as a comma-separated list, nothing else.\n\n    
        {keywords_text}"""

    response = ranker_LLM.invoke(compressor_prompt)
    
    # Extract and clean the macro-keywords
    macro_keywords = [k.strip() for k in response.content.split(',')]
    
    # Update state with macro-keywords
    state["keywords"] = macro_keywords
    return state

def prompter_router(state: PrompterState) -> PrompterState:
    """
    Route the state to the appropriate handler based on the goal.
    The goal can be one of: "writing", "casts", "users"
    """
    state["route"] = state["goal"]
    return state

def writing_tips_handler(state: PrompterState) -> PrompterState:
    """
    Handle writing-related content.
    """
    print("Writing handler")
    return state

def casts_handler(state: PrompterState) -> PrompterState:
    """
    Handle casts-related content.
    """
    return state

def users_handler(state: PrompterState) -> PrompterState:
    """
    Handle users-related content.
    """
    return state

# ---- Graph ---- #
builder = StateGraph(PrompterState)
builder.add_node("fetcher", cast_fetcher)
builder.add_node("extrapolator", topic_extrapolator)
builder.add_node("compressor", topic_compressor)
builder.add_node("prompter_router", prompter_router)
builder.add_node("writing_tips", writing_tips_handler)
builder.add_node("casts_embed", casts_handler)
builder.add_node("users_embed", users_handler)

# Add edges
builder.add_edge(START, "fetcher")
builder.add_edge("fetcher", "extrapolator")
builder.add_edge("extrapolator", "compressor")
builder.add_edge("compressor", "prompter_router")

# Add conditional edges from router based on goal
builder.add_conditional_edges(
    "prompter_router",
    lambda x: x["goal"],
    {
        "writing": "writing_tips",
        "casts": "casts_embed",
        "users": "users_embed"
    }
)

# Add edges to END
builder.add_edge("writing_tips", END)
builder.add_edge("casts_embed", END)
builder.add_edge("users_embed", END)  

# Compile graph
agent_social_prompter = builder.compile()
# Visualize graph
visualize_graph = True
if visualize_graph:
    image = Image(agent_social_prompter.get_graph().draw_mermaid_png())  
    with open("../graphs/social_prompter.png", "wb") as f:
        f.write(image.data)
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [langfuse_handler],
}

messages = agent_social_prompter.invoke({"fid": 4461, "goal": "writing", "casts": [], "keywords": [], "messages": []}, config)
for m in messages['messages']:
    m.pretty_print()

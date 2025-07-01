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

def extract_embed_users(api_response: dict) -> str:
    """
    Extract user information from the API response and return as Markdown text.
    """
    if api_response['status_code'] != 200:
        raise ValueError(f"API response failed: {api_response['body']}")
    
    markdown_output = ""
    users = api_response["body"]
    
    for i, user in enumerate(users, 1):
        user_id = user["user_id"]        
        markdown_output += f"{i}. FID {user_id}\n\n"

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
            elif isinstance(embed, dict) and "metadata" in embed and isinstance(embed["metadata"], dict) and "html" in embed["metadata"]:
                object = embed.get("metadata", {})
                html_content = object.get("html", {})
                if not object:
                    formatted_cast += "\nNo metadata found in embed."
                else:
                    # Extract Open Graph metadata
                    site_name = html_content.get("ogSiteName", "") or html_content.get("siteName", "")
                    title = html_content.get("ogTitle", "") or html_content.get("title", "")
                    description = html_content.get("ogDescription", "") or html_content.get("description", "")
                    embed_text = " - ".join(filter(None, [site_name, title, description]))
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
        "top_k": 3,
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

def get_semantic_user(query: str) -> str:
    """
    Get a list of similar users for a given query.

    Args:
        query: user's query
    """
    url = "https://api.mbd.xyz/v2/farcaster/users/search/semantic"
    payload = {
        "query": query,
        "top_k": 3,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {mbd_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    fetched_results = extract_embed_users(json.loads(response.text))
    return fetched_results


# ---- State of the graph ---- #
class IcePrompterState(MessagesState):
    summary: Optional[str] = None
    conversation_count: Optional[int] = None
    fid : int
    goal: str
    ice : str
    casts: List[str]
    all_keywords: json
    keywords: List[str]


# ---- LLMs ---- #
extrapolator_LLM = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.2)
ranker_LLM = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)
writing_tips_LLM = ChatGroq(model="qwen/qwen3-32b", temperature=0.2)

summarizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
conversational_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.2)
json_filler_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)




# ---- Nodes ---- #
def cast_fetcher(state: IcePrompterState) -> IcePrompterState:
    """
    Fetch casts for the given user ID.
    """
    state["casts"] = get_user_casts(str(state["fid"])) 
    return state

def topic_extrapolator(state: IcePrompterState) -> IcePrompterState:
    """
    Extract keywords from each cast using the LLM and organize them into macro-categories
    with related micro-topics in a JSON format.
    The LLM will analyze both the target post and related post (if present)
    to generate relevant keywords.
    """
    extracted_topics_by_category = {} # Initialize an empty dictionary to store topics by category
    
    for cast in state["casts"]:
        extrapolator_prompt = f"""You are an advanced AI assistant specialized in analyzing social media posts to identify user interests for personalized recommendations. 
        Your goal is to infer concrete, actionable interests from the user's post content and categorize them.

        Instructions:
        - Read the following social media post carefully.
        - Identify the main interest categories (e.g., "Food", "Travel", "Technology").
        - For each main category, identify 1-3 highly specific sub-interests or topics (e.g., "Italian Cuisine", "Hiking in Dolomites", "AI Development").
        - Focus on themes that are:
            - Highly specific and niche, not generic.
            - Indicative of a genuine user interest that could lead to specific recommendations.
            - Actionable for calling recommendation APIs.
        - Return the output in a JSON format where keys are the main categories and values are lists of specific sub-interests.
        - Return ONLY the JSON object, nothing else.

        Example:
        Post content: "I love trying new pasta recipes, especially Roman ones like carbonara! Also planning a trip to Tuscany next summer, maybe rent a villa."
        Output: {{"Food": ["Roman Cuisine", "Pasta Recipes"], "Travel": ["Tuscany Travel", "Villa Rentals"]}}

        Post content:
        {cast}"""

        response = extrapolator_LLM.invoke(extrapolator_prompt)
        
        try:
            # Parse the JSON output from the LLM
            llm_output = json.loads(response.content)
            for category, topics in llm_output.items():
                if category not in extracted_topics_by_category:
                    extracted_topics_by_category[category] = []
                extracted_topics_by_category[category].extend([t.strip().lower() for t in topics])
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON for cast: {cast}")
            print(f"LLM response: {response.content}")
            # Handle cases where LLM might not return perfect JSON
            # You might want to log this or try a regex fallback if needed
            pass
    
    state["all_keywords"] = extracted_topics_by_category
    return state

def topic_compressor(state: IcePrompterState) -> IcePrompterState:
    """
    Compress the list of keywords into a maximum of 5 topics that represent the core themes.
    This function now expects state["keywords"] to be a dictionary with categories and lists of micro-topics.
    """
    if not state["all_keywords"]:
        return state
    
    compressor_prompt = f"""You are an expert conversation topic curator for a social network. 
        Your task is to analyze a structured list of user interests, grouped by main categories, and identify the UP TO 5 most engaging, overarching discussion topics. 
        These topics should be high-level yet clearly derived from the provided specific interests, suitable for sparking a conversation or providing recommendations.
        Strive to represent the breadth of the user's interests by considering topics from different main categories if possible.

        Instructions:
        - Review the provided JSON object where keys are main interest categories and values are lists of specific sub-interests.
        - Identify the 3 to 5 most significant and distinct themes that represent the core interests.
        - Prioritize topics that:
            - Are frequently mentioned or strongly implied across the input interests.
            - Have a clear potential for stimulating conversation or serving as a basis for personalized recommendations.
            - Represent a genuine, strong interest of the user.
            - Ideally, cover interests from different high-level categories if they are prominent.
            
        Return ONLY the selected topics as a comma-separated list, nothing else. 

        Example Input (JSON interests):
        {{
        "Food": ["Roman Cuisine", "Pasta Recipes", "Italian Coffee Making"],
        "Travel": ["Tuscany Travel", "Villa Rentals", "Florence Museums"],
        "Outdoors": ["Hiking Dolomites", "Skiing Alps"],
        "History": ["Ancient History Rome"]
        }}

        Example Output:
        Italian Culinary Experiences, Italian Travel & Culture, Mountain Sports, Roman History

        User interests to analyze:
        {json.dumps(state["all_keywords"], indent=2)}
        """

    response = ranker_LLM.invoke(compressor_prompt)
    
    # Extract and clean the macro-keywords
    keywords = [k.strip() for k in response.content.split(',')]
    # Update state with macro-keywords (now a simple list)
    state["keywords"] = keywords
    return state

def prompter_router(state: IcePrompterState) -> IcePrompterState:
    """
    Route the state to the appropriate handler based on the goal.
    The goal can be one of: "writing", "casts", "users"
    """
    state["route"] = state["goal"]
    return state

def writing_tips_handler(state: IcePrompterState) -> IcePrompterState:
    """
    Handle writing-related content.
    """
    if not state["keywords"] and not state["all_keywords"]:
        return state
    # Generate writing tips based on the keywords
    writing_tips_prompt = f"""You are an expert social media content strategist. 
    Your task is to generate engaging questions that will serve as prompts for the user to create new social media posts.

    Use an introductory sentence, like 'To stimulate your writing for new posts, I've prepared some questions based on your interests.'

    The user has expressed interest in the following core topics:
    {', '.join(state["keywords"])}.
    For EACH of these topics, create ONE thought-provoking question that can directly inspire a social media post. 
    These questions should encourage the user to share their insights or experiences related to the keyword.
    For additional context on the user's broader interests related to the core topics, here is all the detailed information:
    {json.dumps(state["all_keywords"], indent=2)}
    Do not mix topics in a single question; each question should focus on one topic.

    Please present the questions as a bulleted list following the introductory sentence.
    Do not include any other explanations or concluding remarks.

    ---
    **Example 1:**
    **User Keywords:** storytelling, character development
    **All User Interests:**
    {{
    "Writing": {{
        "Storytelling": ["plot twists", "narrative arcs"],
        "Character Development": ["protagonists", "antagonists", "character arcs"]
    }},
    "Genres": {{
        "Fantasy": ["world-building", "magic systems"],
        "Mystery": ["red herrings", "suspense"]
    }}
    }}

    **Desired Output:**
    - What's one surprising plot twist you've loved in a story, and how did it impact your connection to the characters?
    - How do you approach creating a villain whose motivations are as compelling as your hero's?

    ---
    **Example 2:**
    **User Keywords:** digital marketing, SEO, content creation
    **All User Interests:**
    {{
    "Business": {{
        "Digital Marketing": ["social media strategy", "email campaigns"],
        "SEO": ["keyword research", "on-page SEO"],
        "Content Creation": ["blogging", "video marketing"]
    }},
    "Tech": {{
        "Analytics": ["Google Analytics", "data interpretation"]
    }}
    }}

    **Desired Output:**
    - What's the most unexpected challenge you've faced trying to keep up with SEO trends, and how did you adapt your content strategy?
    - If you had to pick just one social media platform for your niche, which would it be and why is it crucial for your digital marketing goals?
    - How do you balance creating engaging video content with the need for strong SEO visibility?

    ---
    """

    response = writing_tips_LLM.invoke(writing_tips_prompt)

    # Extract the questions from the response, only the part after the last <think> tag, if present
    content = response.content
    # Find the index of the last occurrence of '</think>'
    last_think_end_index = content.rfind('</think>')

    if last_think_end_index == -1:
        # If no </think> tag is found, return the entire text
        content_after_think = content
    else:
        # Get all content after the last </think> tag
        content_after_think = content[last_think_end_index + len('</think>'):]

    # Split the content into lines and filter for bullet points
    lines = content_after_think.strip().split('\n')
    writing_tips = []
    for line in lines:
        cleaned_line = line.strip()
        # Check if the line starts with a bullet point (common types: '-', '*')
        # and is not just an empty line or whitespace after stripping.
        if cleaned_line.startswith('- ') or cleaned_line.startswith('* '):
            writing_tips.append(cleaned_line)
        # Also handle cases where a bullet might be followed immediately by text without a space
        elif cleaned_line.startswith('-') or cleaned_line.startswith('*'):
            # Add a space after the bullet if it's missing (for consistent formatting)
            writing_tips.append(cleaned_line[0] + ' ' + cleaned_line[1:].strip())

    tips = "\n\n".join(writing_tips)
    print(f"\nGenerated writing tips:\n\n{tips}")

    return state

def casts_handler(state: IcePrompterState) -> IcePrompterState:
    """
    Handle casts-related content.
    """
    # For each keyword in state["keywords"], call get_semantic_casts and concatenate the results in markdown
    markdown_results = "\n"
    for keyword in state["keywords"]:
        result = get_semantic_casts(keyword)
        markdown_results += f"### Cast for: **{keyword}**\n{result}\n"
    print(markdown_results)
    return state

def users_handler(state: IcePrompterState) -> IcePrompterState:
    """
    Handle users-related content.
    """
    # For each keyword in state["keywords"], call get_semantic_user and concatenate the results in markdown
    markdown_results = "\n"
    for keyword in state["keywords"]:
        result = get_semantic_user(keyword)
        markdown_results += f"### Users for: **{keyword}**\n{result}\n"
    print(markdown_results)
    return state

# ---- Icebreaker ---- #

def icebreaker_router(state: IcePrompterState) -> IcePrompterState:
    """
    Route the state to the appropriate handler based on the goal.
    The goal can be one of: "writing", "casts", "users"
    """
    number_of_casts = len(state["casts"])
    if number_of_casts >= 3:
        state["ice"] = "enough"
    elif number_of_casts < 3 and state["conversation_count"]==3:
        state["ice"] = "now enough"
    else:
        state["ice"] = "not enough"
    return state

def summarize_history(state: IcePrompterState):
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

def json_filler(state: IcePrompterState):
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

def engage_conversation(state: IcePrompterState):
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
builder = StateGraph(IcePrompterState)
builder.add_node('fetcher', cast_fetcher)
builder.add_node('icebreaker_router', icebreaker_router)
builder.add_node('summarizer', summarize_history)
builder.add_node('json_filler', json_filler)
builder.add_node('engage_conversation', engage_conversation)
builder.add_node('extrapolator', topic_extrapolator)
builder.add_node('compressor', topic_compressor)
builder.add_node('prompter_router', prompter_router)
builder.add_node('writing_tips', writing_tips_handler)
builder.add_node('casts_embed', casts_handler)
builder.add_node('users_embed', users_handler)

# Add edges
builder.add_edge(START, "fetcher")
builder.add_edge("fetcher", "icebreaker_router")
builder.add_conditional_edges(
    "icebreaker_router",
    lambda x: x["ice"],
    {
        "enough": "extrapolator",
        "now enough": "prompter_router",
        "not enough": "summarizer",
    }
)
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

builder.add_edge("summarizer", "json_filler")
builder.add_edge("json_filler", "engage_conversation")

# Add edges to END
builder.add_edge("writing_tips", END)
builder.add_edge("casts_embed", END)
builder.add_edge("users_embed", END)  
builder.add_edge("engage_conversation", END)


# Compile graph
agent_social_prompter = builder.compile()
# Visualize graph
visualize_graph = True
if visualize_graph:
    image = Image(agent_social_prompter.get_graph().draw_mermaid_png())  
    with open("../graphs/social_prompter_v2.png", "wb") as f:
        f.write(image.data)
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [langfuse_handler],
}

#messages = agent_social_prompter.invoke({"fid": 6473, "goal": "writing", "casts": [], "all_keywords":{}, "keywords": [], "messages": []}, config)
#for m in messages['messages']:
#    m.pretty_print()

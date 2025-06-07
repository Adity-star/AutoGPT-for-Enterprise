import sys
import os
from typing import TypedDict, List, Dict, Any

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

print("Running Python from:", sys.executable)

from autogpt_core.modules.market_researcher.rebbit_service import RebbitService
from langgraph.graph import StateGraph, END 
from langgraph.graph import add_messages
from dotenv import load_dotenv
import google.generativeai as genai
from config.prompts import get_idea_generation_prompt
import json


load_dotenv()

class MarketResearchState(TypedDict):
    trending_posts: List[Dict[str, Any]]
    idea_list: List[Dict[str, Any]]

# Wrapper function to integrate RedditService with graph
def get_trending_industries(state: MarketResearchState) -> MarketResearchState:
    reddit = RebbitService()
    top_posts = reddit.get_business_trending_posts(limit=10)
    return {"trending_posts": top_posts, "idea_list": []}

def generate_idea_list(state: MarketResearchState) -> MarketResearchState:
    posts = state["trending_posts"]
    topics = [post['title'] for post in posts]

    prompt = get_idea_generation_prompt("\n".join(topics))

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    try:
        ideas = json.loads(response.text)
    except json.JSONDecodeError:
        print("Could not parse JSON, response was:")
        print(response.text)
        ideas = {"ideas": []}

    return {"idea_list": ideas.get("ideas", [])}

graph = StateGraph(state_schema=MarketResearchState)

graph.add_node("get_trending_industries", get_trending_industries)
graph.add_node("generate_idea_list", generate_idea_list)

graph.set_entry_point("get_trending_industries")
graph.add_edge("get_trending_industries", "generate_idea_list")

market_research_agent = graph.compile()
result = market_research_agent.invoke({})

print(result)
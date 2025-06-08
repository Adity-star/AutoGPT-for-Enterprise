import sys
import os
import json
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END 
from langgraph.graph import add_messages
import google.generativeai as genai

from modules.market_researcher.rebbit_service import RebbitService
from config.prompts import get_idea_generation_prompt
from utils.helper import safe_parse_ideas
from utils.logger import logging
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    print(f"🧠 Sending prompt:\n{prompt}")

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    ideas = safe_parse_ideas(response.text)

    return {
        "trending_posts": state.get("trending_posts", []),
        "idea_list": ideas
    }

graph = StateGraph(state_schema=MarketResearchState)

graph.add_node("get_trending_industries", get_trending_industries)
graph.add_node("generate_idea_list", generate_idea_list)

graph.set_entry_point("get_trending_industries")
graph.add_edge("get_trending_industries", "generate_idea_list")

market_research_agent = graph.compile()
result = market_research_agent.invoke({})

print("✅ Final result:")
print(json.dumps(result, indent=2))
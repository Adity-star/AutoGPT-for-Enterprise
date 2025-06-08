import sys
import os
import json
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END 
from langgraph.graph import add_messages
import google.generativeai as genai


from modules.market_researcher.rebbit_service import RebbitService
from autogpt_core.config.prompts import get_idea_generation_prompt
from utils.helper import safe_parse_ideas
from modules.market_researcher.support import analyze_ideas_with_trends,export_graph_to_mermaid
from utils.logger import logging
from dotenv import load_dotenv



load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Running Market Research Agent")


class MarketResearchState(TypedDict):
    trending_posts: List[Dict[str, Any]]
    idea_list: List[Dict[str, Any]]
    analyse_demands: List[Dict[str, Any]]



# Wrapper function to integrate RedditService with graph
def get_trending_industries(state: MarketResearchState) -> MarketResearchState:
    reddit = RebbitService()
    top_posts = reddit.get_business_trending_posts(limit=10)
    return {"trending_posts": top_posts}



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


def analyze_demand(state: MarketResearchState) -> MarketResearchState:
    # Step 1: Analyze ideas with Google Trends
    state = analyze_ideas_with_trends(state)

    ideas_data = state.get("idea_trend_scores", [])

    # Step 2: Create Gemini model
    model = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-1.5-pro"

    # Step 3: Analyze each idea with Gemini
    for idea_data in ideas_data:
        prompt = (
            f"Evaluate the market demand for the business idea: {idea_data['idea']}.\n"
            f"Google Trends score (0-100): {idea_data['trend_score']}\n"
            "Provide a brief analysis and assign a demand score from 1 to 10."
        )

        try:
            response = model.generate_content(prompt)
            idea_data["gpt_analysis"] = response.text
        except Exception as e:
            idea_data["gpt_analysis"] = f"Error generating analysis: {str(e)}"

    state["demand_analysis"] = ideas_data
    return state




graph = StateGraph(state_schema=MarketResearchState)

graph.add_node("get_trending_industries", get_trending_industries)
graph.add_node("generate_idea_list", generate_idea_list)
graph.add_node("analyze_demand", analyze_demand) 

graph.set_entry_point("get_trending_industries")

graph.add_edge("get_trending_industries", "generate_idea_list")
graph.add_edge("generate_idea_list", "analyze_demand")


market_research_agent = graph.compile()
result = market_research_agent.invoke({})

print("✅ Final result:")
print(json.dumps(result, indent=2))
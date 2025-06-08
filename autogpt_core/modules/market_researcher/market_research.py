import sys
import os
import json
import time
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END 
from langgraph.graph import add_messages
import google.generativeai as genai


from modules.market_researcher.rebbit_service import RebbitService
from autogpt_core.config.prompts import get_idea_generation_prompt
from utils.helper import safe_parse_ideas
from modules.market_researcher.support import analyze_ideas_with_trends,export_graph_to_mermaid,search_competitors
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
    analyse_competition: List[Dict[str, Any]]
    analyse_unit_economics: List[Dict[str, Any]]
    score_ideas: List[Dict[str, Any]]
    validate_idea: List[Dict[str, Any]]
    select_idea: List[Dict[str, Any]]



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


def analyze_competition(state: MarketResearchState) -> MarketResearchState:
    demand_analysis = state.get("demand_analysis", [])

    # Initialize Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

    for idea_data in demand_analysis:
        idea = idea_data.get("idea")
        if not idea:
            continue

        # Search for competitors
        competitors = search_competitors(idea)

        # Format competitor list
        competitors_str = "\n".join(
            f"- {c['title']}" if isinstance(c, dict) else f"- {c}" 
            for c in competitors
        )

        # Prompt for Gemini analysis
        prompt = (
            f"Business Idea: {idea}\n"
            f"Competitors found via search:\n{competitors_str or 'None found'}\n\n"
            "Analyze the current market competition level based on:\n"
            "- Direct and indirect competitors\n"
            "- Market saturation\n"
            "- Entry barriers\n\n"
            "Give a brief assessment and rate the competition level on a scale of 1 (low) to 10 (high)."
        )

        try:
            response = model.generate_content(prompt)
            idea_data["competition_analysis"] = {
                "competitors": competitors,
                "gpt_summary": response.text
            }

        except Exception as e:
            idea_data["competition_analysis"] = {
                "competitors": competitors,
                "gpt_summary": f"Error during Gemini analysis: {str(e)}"
            }

        time.sleep(1.5)  # Respect API limits

    return {
        **state,
        "competition_analysis": demand_analysis
    }


def analyse_unit_economics(state: MarketResearchState)-> MarketResearchState:
    ideas = state.get("demand_analysis",[])
    analyzed = []

    model = genai.GenerativeModel("gemini-1.5-flash")

    for idea in ideas:
        prompt = f"""
        For the business idea: {idea['idea']}, provide a breakdown of unit economics.
        Include estimates for:
        - Customer Acquisition Cost (CAC)
        - Expected Revenue per Customer
        - Gross Margin %
        - Break-even timeline
        - One-line business model summary

        Use realistic assumptions based on the current market.
        """

    try:
            response = model.generate_content(prompt)
            idea["unit_economics"] = response.text
    except Exception as e:
            idea["unit_economics"] = f"Error during Gemini analysis: {str(e)}"

    analyzed.append(idea)

    time.sleep(1.5)  # rate limit respect

    return {
        **state,
        "unit_economics_analysis": analyzed
    }

def score_ideas(state: MarketResearchState) -> MarketResearchState:
    ideas = state.get("unit_economics_analysis", [])
    scored_ideas = []

    model = genai.GenerativeModel("gemini-1.5-flash")

    for idea_data in ideas:
            prompt = f"""
            Score the following startup idea using the criteria below (each 1–10):
            - Market Demand (based on Trends + analysis)
            - Competition Level (lower = better)
            - Unit Economics (based on CAC, margin, model)

            Respond with:
            - Scores for each
            - A final weighted score (out of 100)
            - One-line summary of why this idea is promising or risky

            ---
            Business Idea: {idea_data['idea']}
            Trend Score: {idea_data.get('trend_score')}
            Demand Analysis: {idea_data.get('gpt_analysis')}
            Competition Analysis: {idea_data.get('competition_analysis')}
            Unit Economics: {idea_data.get('unit_economics')}
            """

            try:
                response = model.generate_content(prompt)
                idea_data["scoring_summary"] = response.text
            except Exception as e:
                idea_data["scoring_summary"] = f"Error: {str(e)}"

            scored_ideas.append(idea_data)

    return {
            **state,
            "scored_ideas": scored_ideas
        }
 
def validate_idea(state: MarketResearchState) -> MarketResearchState:
    ideas = state.get("scored_ideas", [])
    validated_ideas = []

    model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"
    
    for idea in ideas:
            prompt = f"""
            You're a venture capitalist evaluating this startup pitch:

            Idea: {idea['idea']}
            Summary: {idea.get('scoring_summary', '')}

            Give feedback:
            - Is this worth exploring further?
            - What assumptions need to be tested?
            - Rate the validation readiness (1-10)
            """

            try:
                response = model.generate_content(prompt)
                idea["validation_feedback"] = response.text
            except Exception as e:
                idea["validation_feedback"] = f"Error: {e}"

            validated_ideas.append(idea)

    return {
            **state,
            "validated_ideas": validated_ideas
        }

def select_idea(state: MarketResearchState) -> MarketResearchState:
    ideas = state.get("validated_ideas", [])
     
    def extract_score(idea):
            try:
                # Let's assume final score is in the summary as "Score: 82/100"
                import re
                match = re.search(r"score.*?(\d{2,3})[\/100]?", idea.get("scoring_summary", ""), re.IGNORECASE)
                return int(match.group(1)) if match else 0
            except:
                return 0

    best_idea = max(ideas, key=extract_score, default=None)

    return {
            **state,
            "best_business_idea": best_idea
        }


graph = StateGraph(state_schema=MarketResearchState)

graph.add_node("get_trending_industries", get_trending_industries)
graph.add_node("generate_idea_list", generate_idea_list)
graph.add_node("analyze_demand", analyze_demand) 
graph.add_node("analyse_competition",analyze_competition)
graph.add_node("analyse_unit_economics",analyse_unit_economics)
graph.add_node("score_ideas", score_ideas)
graph.add_node("validate_top_ideas", validate_idea)
graph.add_node("select_best_idea", select_idea)



graph.set_entry_point("get_trending_industries")

graph.add_edge("get_trending_industries", "generate_idea_list")
graph.add_edge("generate_idea_list", "analyze_demand")
graph.add_edge("analyze_demand","analyse_competition")
graph.add_edge("analyse_competition","analyse_unit_economics")
graph.add_edge("analyze_unit_economics", "score_ideas")
graph.add_edge("score_ideas", "validate_top_ideas")
graph.add_edge("validate_top_ideas", "select_best_idea")
graph.add_edge("select_best_idea", "output_idea")



market_research_agent = graph.compile()
result = market_research_agent.invoke({})

print("✅ Final result:")
print(json.dumps(result, indent=2))
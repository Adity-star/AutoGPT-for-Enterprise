from pytrends.request import TrendReq
from typing import Dict, Any
import logging
import requests
from bs4 import BeautifulSoup
import re
import time 
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


MarketResearchState = Dict[str, Any]

def analyze_ideas_with_trends(state: MarketResearchState) -> MarketResearchState:
    ideas = state.get("idea_list", [])
    pytrends = TrendReq()
    trend_scores = []

    for idea in ideas:
        keyword = idea.get("title", idea) if isinstance(idea, dict) else idea
        try:
            pytrends.build_payload([keyword], timeframe='today 12-m')
            data = pytrends.interest_over_time()
            score = round(data[keyword].mean(), 2) if not data.empty else 0.0
        except Exception as e:
            logger.error(f"Error fetching trends for {keyword}: {e}")
            score = 0.0

        trend_scores.append({"idea": keyword, "trend_score": score})

    return {
        **state,
        "idea_trend_scores": trend_scores
    }


# def export_graph_to_mermaid(graph: StateGraph) -> str:
#     lines = ["graph TD"]

#     # graph.edges is likely a set of (source, destination) tuples
#     for edge in graph.edges:
#         if isinstance(edge, tuple) and len(edge) == 2:
#             source, destination = edge
#             lines.append(f"    {source} --> {destination}")

#     return "\n".join(lines)


def search_competitors(idea, max_results=5):
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            logger.error("SERPAPI_API_KEY environment variable is not set")
            return []

        logger.info(f"Searching competitors for idea: {idea}")

        params = {
            "q": f"{idea} competitors",
            "num": max_results,
            "engine": "google",
            "api_key": api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            logger.error(f"SerpAPI returned an error: {results['error']}")
            return []

        competitors = []
        for result in results.get("organic_results", [])[:max_results]:
            title = result.get("title")
            if title:
                competitors.append(title.strip())

        logger.info(f"Found {len(competitors)} competitors")
        return competitors

    except Exception as e:
        logger.error(f"SerpAPI error: {str(e)}")
        return []
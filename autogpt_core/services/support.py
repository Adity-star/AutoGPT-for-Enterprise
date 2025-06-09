from pytrends.request import TrendReq
from typing import Dict, Any
import logging
from langgraph.graph import StateGraph,END
import requests
from bs4 import BeautifulSoup
import re
import time 
from serpapi import GoogleSearch
import os

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


def export_graph_to_mermaid(graph: StateGraph) -> str:
    lines = ["graph TD"]

    # graph.edges is likely a set of (source, destination) tuples
    for edge in graph.edges:
        if isinstance(edge, tuple) and len(edge) == 2:
            source, destination = edge
            lines.append(f"    {source} --> {destination}")

    return "\n".join(lines)


def search_competitors(idea, max_results=5):
    try:
        search = GoogleSearch({
            "q": f"{idea} competitors",
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": max_results
        })
        results = search.get_dict()
        competitors = []

        for result in results.get("organic_results", [])[:max_results]:
            title = result.get("title")
            if title:
                competitors.append(title.strip())

        return competitors

    except Exception as e:
        print(f"[SerpAPI error]: {e}")
        return []







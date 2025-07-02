from pytrends.request import TrendReq
from typing import Dict, Any
import logging
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
from autogpt_core.core.secrets import secrets

load_dotenv()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


MarketResearchState = Dict[str, Any]



def search_competitors(idea, max_results=5):
    try:
        api_key = secrets.SERPAPI_API_KEY
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
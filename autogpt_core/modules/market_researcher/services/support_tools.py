from typing import Dict, Any
import logging
from serpapi import GoogleSearch
from dotenv import load_dotenv
from autogpt_core.core.secrets import secrets
import json 
import re
import asyncio
from typing import Optional

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
    

def strip_json_markdown(text: str) -> str:
    """Strips Markdown-style code blocks from the response."""
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()


async def safe_parse_json_from_llm(llm_service, prompt: str, retries: int = 3, delay: float = 2.0):
    last_response = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"LLM JSON parse attempt {attempt}")
            response_text = await llm_service.chat(prompt)
            last_response = response_text

            if not response_text or not response_text.strip():
                raise ValueError("Empty response received from LLM")

            logger.debug(f"Response type: {type(response_text)}")
            logger.debug(f"Raw response (first 300 chars): {repr(response_text[:300])}")

            clean_text = strip_json_markdown(response_text)
            data = json.loads(clean_text)

            if not isinstance(data, dict) or "ideas" not in data:
                raise ValueError("Parsed JSON is missing 'ideas' key")

            return data

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed due to unexpected error: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {retries} attempts failed. Last response was:\n{last_response}")
                return None
            

async def safe_parse_json_response(llm_service, prompt: str, retries: int = 3, delay: float = 2.0) -> Optional[Dict[str, Any]]:
    last_response = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"LLM JSON parse attempt {attempt}")
            response_text = await llm_service.chat(prompt)
            last_response = response_text

            if not response_text or not response_text.strip():
                raise ValueError("Empty response received from LLM")

            clean_text = strip_json_markdown(response_text)
            data = json.loads(clean_text)

            if not isinstance(data, dict):
                raise ValueError("Parsed response is not a JSON object")

            return data

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed due to unexpected error: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {retries} attempts failed. Last response was:\n{last_response}")
                return None

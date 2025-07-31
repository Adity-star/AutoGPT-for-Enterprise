import json
from utils.logger import logger
from autogpt_core.core.llm_service import LLMService
import re

def extract_json_list(text: str):
    
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return []
    return []

async def extract_keywords(summary: str, max_keywords: int = 10) -> list[str]:
    logger.info("Extracting keywords from research summary...")

    prompt = f"""
    You're an expert SEO strategist.

    Extract the top {max_keywords} relevant keywords or key phrases from the text below.
    Focus on phrases people might search for.

    Research Summary:
    \"\"\"{summary}\"\"\"

    Return only a JSON array of strings like:
    ["keyword1", "keyword2", "long tail keyword 3", ...]
    """

    try:
        llm_service = LLMService()
        response = await llm_service.chat(prompt)
        logger.info(f"LLM raw response: {response!r}")
        try:
            keywords = json.loads(response.strip())
        except Exception:
            keywords = extract_json_list(response)
        return keywords
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return []


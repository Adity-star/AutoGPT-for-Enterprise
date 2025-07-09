 # Keyword extraction + injection

from autogpt_core.modules.blog_writer.blog_services.blog_state import BlogWriterAgentState
from autogpt_core.core.llm_service import LLMService
from utils.logger import logger


async def extract_keywords(summary: str, max_keywords: int = 10) -> list[str]:
    """
    Extract SEO-relevant keywords from a research summary.
    """
    logger.info("Extracting keywords from research summary...")

    prompt = f"""
    You're an expert SEO strategist.

    Extract the top {max_keywords} relevant keywords or key phrases from the text below.
    Focus on phrases people might search for.

    Research Summary:
    \"\"\"{summary}\"\"\"

    Return only a Python list of strings like:
    ["keyword1", "keyword2", "long tail keyword 3", ...]
    """

    try:
        response = await LLMService.sync_chat(prompt)
        keywords = eval(response.strip())  # If you're confident in format control
        return keywords
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return []
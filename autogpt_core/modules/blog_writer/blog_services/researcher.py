import httpx
from utils.logger import logger
from autogpt_core.modules.blog_writer.blog_services.blog_state import BlogWriterAgentState
from autogpt_core.core.secrets import secrets
from autogpt_core.core.llm_service import LLMService


async def fetch_serp_results(query: str, num_results: int = 5) -> str:
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": secrets.SERP_API_KEY,
        "num": num_results,
        "engine": "google",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            snippets = [
                result["snippet"] 
                for result in data.get("organic_results", []) 
                if "snippet" in result
            ]
            return "\n".join(snippets[:num_results])
        except Exception as e:
            logger.error(f"SERP API error: {e}")
            return ""


async def run_blog_research(state: BlogWriterAgentState) -> BlogWriterAgentState:
    if not state.idea_data:
        raise ValueError("Missing idea data for research.")

    idea = state.idea_data.get("idea", "")
    logger.info(f"Researching factual data for: {idea}")

    serp_snippets = await fetch_serp_results(idea)
    if not serp_snippets:
        serp_snippets = "No factual SERP data found. Continue with LLM knowledge only."

    prompt = f"""
    You're writing a blog post based on this startup idea:
    "{idea}"

    Here are factual search results from SERP:
    {serp_snippets}

    Using this, write a concise summary including:
    - Market Overview
    - Customer Pain Points
    - Current Trends
    - Unique Opportunity Angle
    """

    try:
        llm_service = LLMService()  # Instantiate service (adjust if needed)
        summary = await llm_service.chat(prompt)
        summary = summary.strip()
    except Exception as e:
        logger.error(f"LLM failed to generate research summary: {e}")
        summary = "LLM research summary unavailable."

    # Return updated copy of state with new research summary
    return state.model_copy(update={"research_summary": summary})

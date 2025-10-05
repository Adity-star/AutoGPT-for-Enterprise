from dotenv import load_dotenv
load_dotenv()
from typing import Optional, Dict, Any
import asyncio

from autogpt_core.utils.logger import logging
from autogpt_core.modules.market_researcher.graph import get_or_generate_market_research_idea
from autogpt_core.modules.market_researcher.state import MarketResearchState, AnalysisConfig
from backend.logging_config import setup_logging
from backend.error_handler import handle_exception, ValidationError, APIError
from autogpt_core.modules.market_researcher.nodes import parallel_analysis
from autogpt_core.planner import AgentPlanner

# Configure logging with timestamped files
logger = setup_logging("market_research_api")

async def run_market_research(
    user_idea: Optional[str] = None,
    max_retries: Optional[int] = 2,
    batch_size: Optional[int] = 3,
    timeout: Optional[int] = 30,
    enable_caching: Optional[bool] = True,
    cache_ttl_minutes: Optional[int] = 60
) -> Dict[str, Any]:
    """
    Runs the market research process.

    Args:
        user_idea: The user's business idea.
        max_retries: The maximum number of retries for each step.
        batch_size: The batch size for parallel analysis.
        timeout: The timeout for each step.
        enable_caching: Whether to enable caching.
        cache_ttl_minutes: The cache TTL in minutes.

    Returns:
        The result of the market research.
    """
    try:
        logger.info("Received market research request")

        # Validate parameters
        if max_retries < 0:
            raise ValidationError("max_retries must be non-negative", {"max_retries": max_retries})
        if batch_size < 1:
            raise ValidationError("batch_size must be >= 1", {"batch_size": batch_size})

        # Create config and state
        config = AnalysisConfig(
            max_retries=max_retries,
            batch_size=batch_size,
            timeout=timeout,
            enable_caching=enable_caching,
            cache_ttl_minutes=cache_ttl_minutes
        )

        state = MarketResearchState()
        state.user_idea = user_idea
        state.config = config

        # Call caching wrapper with state
        result = await get_or_generate_market_research_idea(state)

        return {"status": "success", "best_idea": result.get("best_business_idea", {})}

    except Exception as e:
        error_details = handle_exception(e)
        return {"status": "error", "error": error_details}
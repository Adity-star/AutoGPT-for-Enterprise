from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio

from utils.logger import logging
from modules.market_researcher.graph import get_or_generate_market_research_idea
from modules.market_researcher.state import MarketResearchState, AnalysisConfig
from app.logging_config import setup_logging
from app.error_handler import handle_exception, ValidationError, APIError
from modules.market_researcher.nodes import parallel_analysis
from dotenv import load_dotenv
load_dotenv()

# Configure logging with timestamped files
logger = setup_logging("market_research_api")

app = FastAPI(title="Market Research Agent API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MarketResearchRequest(BaseModel):
    user_idea: Optional[str] = None
    max_retries: Optional[int] = 2
    batch_size: Optional[int] = 3
    timeout: Optional[int] = 30
    enable_caching: Optional[bool] = True
    cache_ttl_minutes: Optional[int] = 60

class MarketResearchResponse(BaseModel):
    status: str
    best_idea: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@app.post("/api/research", response_model=MarketResearchResponse)
async def run_market_research(request: MarketResearchRequest):
    try:
        logger.info("Received market research request")

        # Validate parameters
        if request.max_retries < 0:
            raise ValidationError("max_retries must be non-negative", {"max_retries": request.max_retries})
        if request.batch_size < 1:
            raise ValidationError("batch_size must be >= 1", {"batch_size": request.batch_size})

        # Create config and state
        config = AnalysisConfig(
            max_retries=request.max_retries,
            batch_size=request.batch_size,
            timeout=request.timeout,
            enable_caching=request.enable_caching,
            cache_ttl_minutes=request.cache_ttl_minutes
        )

        state = MarketResearchState()
        state.user_idea = request.user_idea
        state.config = config

        # Call caching wrapper with state
        result = await get_or_generate_market_research_idea(state)

        best_idea = result.get("best_business_idea", {})

        return MarketResearchResponse(
            status="success",
            best_idea=best_idea
        )

    except Exception as e:
        error_details = handle_exception(e)
        return MarketResearchResponse(
            status="error",
            error=error_details
        )


@app.post("/api/analyze_idea", response_model=MarketResearchResponse)
async def analyze_single_idea(request: MarketResearchRequest):
    try:
        if not request.user_idea:
            raise ValidationError("user_idea is required for single idea analysis", {})

        config = AnalysisConfig(
            max_retries=request.max_retries,
            batch_size=request.batch_size,
            timeout=request.timeout,
            enable_caching=request.enable_caching,
            cache_ttl_minutes=request.cache_ttl_minutes
        )

        # Inject user idea and config into state
        state = MarketResearchState(
            user_idea=request.user_idea,
            config=config
        )

        # This now handles analysis + validation (skipping generation)
        result = await get_or_generate_market_research_idea(state)

        return MarketResearchResponse(
            status="success",
            best_idea=result.get("best_business_idea", {})
        )

    except Exception as e:
        error_details = handle_exception(e)
        return MarketResearchResponse(
            status="error",
            error=error_details
        )



@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

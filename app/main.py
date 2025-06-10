from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from utils.logger import logging
from autogpt_core.modules.market_researcher.market_research import run_market_research_agent, AnalysisConfig
from app.logging_config import setup_logging
from app.error_handler import handle_exception, ValidationError, APIError

# Configure logging with timestamped files
logger = setup_logging("market_research_api")

app = FastAPI(title="Market Research Agent API")

# Allow your frontend (Streamlit or others) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MarketResearchRequest(BaseModel):
    max_retries: Optional[int] = 2
    batch_size: Optional[int] = 3
    timeout: Optional[int] = 30
    enable_caching: Optional[bool] = True
    cache_ttl_minutes: Optional[int] = 60

class MarketResearchResponse(BaseModel):
    status: str
    best_idea: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

@app.post("/api/research", response_model=MarketResearchResponse)
async def run_market_research(request: MarketResearchRequest):
    try:
        logger.info("Starting market research request")
        
        # Validate request parameters
        if request.max_retries < 0:
            raise ValidationError(
                "max_retries must be non-negative",
                details={"max_retries": request.max_retries}
            )
        
        if request.batch_size < 1:
            raise ValidationError(
                "batch_size must be at least 1",
                details={"batch_size": request.batch_size}
            )
        
        # Create config from request
        config = AnalysisConfig(
            max_retries=request.max_retries,
            batch_size=request.batch_size,
            timeout=request.timeout,
            enable_caching=request.enable_caching,
            cache_ttl_minutes=request.cache_ttl_minutes
        )
        
        # Run the market research agent
        result = await run_market_research_agent(config)
        
        # Extract the best idea
        best_idea = result.get('best_business_idea', {}).get('idea') if result else None
        
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

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

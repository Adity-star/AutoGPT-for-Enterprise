from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from autogpt_core.modules.market_researcher.market_research import market_research_agent

app = FastAPI(title="Market Research Agent API")

# Allow your frontend (Streamlit or others) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, specify domains explicitly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MarketResearchRequest(BaseModel):
    industry: str
    keywords: List[str]

class MarketResearchResponse(BaseModel):
    status: str
    data: dict
    message: Optional[str] = None

@app.post("/api/research", response_model=MarketResearchResponse)
async def run_market_research(request: MarketResearchRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            lambda: market_research_agent.invoke({
                "industry": request.industry,
                "keywords": request.keywords
            })
        )
        return MarketResearchResponse(
            status="success",
            data=result,
            message="Market research completed successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running market research: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

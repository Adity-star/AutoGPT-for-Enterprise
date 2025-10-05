from backend.api_schema import ResearchInput
from backend.main import run_market_research

async def run_market_agent(input: ResearchInput):
    input_dict = input.dict()
    return await run_market_research(**input_dict)
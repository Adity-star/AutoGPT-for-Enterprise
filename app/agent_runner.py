from app.api_schema import ResearchInput
from modules.market_researcher.market_research import run_market_agent as langchain_runner

def run_market_agent(input: ResearchInput):
    input_dict = input.dict()
    return langchain_runner(input_dict)

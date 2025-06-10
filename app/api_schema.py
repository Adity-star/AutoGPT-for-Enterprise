from modules.market_researcher import market_research  
class ResearchInput:
    def __init__(self, keywords, industry):
        self.keywords = keywords
        self.industry = industry

def run_market_agent(input: ResearchInput):
    
    initial_state = {
        "keywords": input.keywords,
        "industry": input.industry
    }

    # Pass this state to your agent graph
    result = market_research.run_market_research_agent(initial_state)
    return result


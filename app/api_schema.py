from modules.market_researcher import market_research  
# Define ResearchInput here if not available elsewhere
class ResearchInput:
    def __init__(self, keywords, industry):
        self.keywords = keywords
        self.industry = industry

def run_market_agent(input: ResearchInput):
    # Prepare initial state with input data
    initial_state = {
        "keywords": input.keywords,
        "industry": input.industry
    }

    # Pass this state to your agent graph
    result = market_research.invoke(initial_state)
    return result


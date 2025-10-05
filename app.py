import asyncio
from backend.agent_runner import run_market_agent
from backend.api_schema import ResearchInput

async def main():
    """Main entry point for the market research agent"""
    try:
        # Example of running the market research agent with a user idea
        user_idea = "AI-powered personal finance assistant"
        results = run_market_agent(ResearchInput(user_idea=user_idea))
        print("Market Research Results:")
        import json
        print(json.dumps(results, indent=2))

        # Example of running the market research agent to generate a new idea
        results = run_market_agent(ResearchInput())
        print("Market Research Results (New Idea):")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"Error running market research agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())
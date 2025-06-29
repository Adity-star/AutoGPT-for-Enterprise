from autogpt_core.modules.market_researcher.nodes import (
    get_trending_industries,
    generate_idea_list,
    parallel_analysis,
    combine_and_score,
    validate_and_select,
    store_results_to_file,
)
from datetime import datetime
from langgraph.graph import StateGraph
import asyncio
from autogpt_core.modules.market_researcher.state import AnalysisConfig, MarketResearchState
from typing import Dict, Any, Optional
from utils.logger import logger
from autogpt_core.modules.market_researcher.nodes import AsyncResilientAnalyzer
from utils.idea_memory import load_ideas_from_db, save_idea_to_db, init_db
from datetime import datetime, timedelta


analyzer = AsyncResilientAnalyzer(AnalysisConfig())

# Create the workflow graph
def create_market_research_graph() -> StateGraph:
    """Create and configure the market research workflow graph"""
    
    graph = StateGraph(MarketResearchState)
    
    # Add nodes
    graph.add_node("get_trending_industries", get_trending_industries)
    graph.add_node("generate_idea_list", generate_idea_list)
    graph.add_node("parallel_analysis_2", parallel_analysis)
    graph.add_node("combine_and_score", combine_and_score)
    graph.add_node("validate_and_select", validate_and_select)
    graph.add_node("store_results_to_file", store_results_to_file)
    
    # Define workflow
    graph.set_entry_point("get_trending_industries")
    graph.add_edge("get_trending_industries", "generate_idea_list")
    graph.add_edge("generate_idea_list", "parallel_analysis_2")
    graph.add_edge("parallel_analysis_2", "combine_and_score")
    graph.add_edge("combine_and_score", "validate_and_select")
    graph.add_edge("validate_and_select", "store_results_to_file")
    
    return graph

def get_recent_idea(hours=24) -> Optional[Dict[str, Any]]:
    """Load idea from DB if created within last `hours`"""
    ideas = load_ideas_from_db(limit=1)
    if not ideas:
        return None
    idea = ideas[0]
    created_at_str = idea.get("created_at")
    if created_at_str:
        created_at = datetime.fromisoformat(created_at_str)
        if datetime.utcnow() - created_at < timedelta(hours=hours):
            return idea
    return None

async def run_market_research_agent(config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
    """Main function to run the market research agent"""
    
    if config is None:
        config = AnalysisConfig(
            max_retries=2,
            batch_size=3,
            timeout=30,
            enable_caching=True,
            cache_ttl_minutes=60
        )
    
    # Update global analyzer config
    global analyzer
    analyzer.config = config
    
    # Create initial state
    initial_state = MarketResearchState(
        config=config,
        start_time=datetime.now()
    )
    
    # Create and compile graph
    graph = create_market_research_graph()
    market_research_agent = graph.compile()

    # Initialize DB before saving to ensure tables exist
    from utils.idea_memory import init_db  # Replace with your actual module
    init_db()

    
    try:
        logger.info("Starting Market Research...")
        result = await market_research_agent.ainvoke(initial_state)
        
        # Only return the best business idea with its complete analysis
        best_idea = result.get("best_business_idea")
        if best_idea:
            return {
                "best_business_idea": {
                    "idea": best_idea.get("idea", ""),
                    "trend_score": best_idea.get("trend_score", 0),
                    "demand_analysis": best_idea.get("demand_analysis", ""),
                    "demand_score": best_idea.get("demand_score", ""),
                    "competition_analysis": best_idea.get("competition_analysis", ""),
                    "competition_score": best_idea.get("competition_score", ""),
                    "unit_economics": best_idea.get("unit_economics", ""),
                    "economics_score": best_idea.get("economics_score", ""),
                    "final_score": best_idea.get("final_score", 0),
                    "scoring_breakdown": best_idea.get("scoring_breakdown", ""),
                    "validation_score": best_idea.get("validation_score", 0),
                    "recommendation": best_idea.get("recommendation", ""),
                    "validation_summary": best_idea.get("validation_summary", "")
                }
            }
        else:
            return {"best_business_idea": None}
            
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {"error": str(e)}
    
async def get_or_generate_market_research_idea(state: MarketResearchState, hours=24) -> Dict[str, Any]:
    init_db()

    if state.user_idea:
        # Skip generation — run direct validation pipeline
        logger.info("User provided idea. Running direct analysis and validation...")

        graph = StateGraph(MarketResearchState)
        graph.add_node("parallel_analysis", parallel_analysis)
        graph.add_node("validate_and_select", validate_and_select)
        graph.add_node("store_results_to_file", store_results_to_file)

        graph.set_entry_point("parallel_analysis")
        graph.add_edge("parallel_analysis", "validate_and_select")
        graph.add_edge("validate_and_select", "store_results_to_file")

        compiled_graph = graph.compile()

        state.idea_list = [{"idea": state.user_idea}]
        state.start_time = datetime.utcnow()

        final_state = await compiled_graph.ainvoke(state)
        return {
            "best_business_idea": final_state.get("best_business_idea",{})
        }

    # No user idea — check if recent idea exists in DB
    ideas = load_ideas_from_db(limit=1)
    if ideas:
        idea = ideas[0]
        created_at_str = idea.get("created_at")
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str)
            if datetime.utcnow() - created_at < timedelta(hours=hours):
                logger.info("Using cached idea from DB (within last 24 hours).")
                return {
                    "best_business_idea": idea
                }

    # Run full market research agent
    logger.info("No recent idea found. Running full market research agent...")
    result = await run_market_research_agent(state.config)

    if result and "best_business_idea" in result:
        best_idea = result["best_business_idea"]
        best_idea["created_at"] = datetime.utcnow().isoformat()
        save_idea_to_db(best_idea)
        return result

    return {"best_business_idea": None}
    

    

if __name__ == "__main__":
    idea = asyncio.run(get_or_generate_market_research_idea())
    print("Business Idea:", idea)
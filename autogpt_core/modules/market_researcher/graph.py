from .nodes import (
    get_trending_industries,
    generate_idea_list,
    parallel_analysis,
    combine_and_score,
    validate_and_select,
    store_results_to_file,

)
from datetime import datetime
from .state import MarketResearchState
from langgraph.graph import StateGraph
import asyncio
from .state import AnalysisConfig
from typing import Dict, Any, Optional
from utils.logger import logger
from autogpt_core.modules.market_researcher.nodes import AsyncResilientAnalyzer


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
    
    try:
        logger.info("Starting Market Research...")
        result = await market_research_agent.ainvoke(initial_state)
        
        # Only return the best business idea with its complete analysis
        best_idea = result.best_business_idea
        if best_idea:
            return {
                "best_business_idea": {
                    "idea": best_idea.get("idea", ""),
                    "trend_score": best_idea.get("trend_score", 0),
                    "demand_analysis": best_idea.get("demand_analysis", ""),
                    "competition_analysis": best_idea.get("competition_analysis", ""),
                    "unit_economics": best_idea.get("unit_economics", ""),
                    "demand_score": best_idea.get("demand_score", 0),
                    "competition_score": best_idea.get("competition_score", 0),
                    "economics_score": best_idea.get("economics_score", 0),
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

if __name__ == "__main__":
    asyncio.run(run_market_research_agent())
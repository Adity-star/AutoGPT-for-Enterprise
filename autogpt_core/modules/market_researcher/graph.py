from autogpt_core.modules.market_researcher.nodes import (
    get_trending_industries,
    generate_idea_list,
    parallel_analysis,
    combine_and_score,
    validate_and_select,
    store_results_to_file,
    analyze_ideas_with_trends
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

def create_market_research_graph() -> StateGraph:

    graph = StateGraph(MarketResearchState)

    graph.add_node("get_trending_industries", get_trending_industries)
    graph.add_node("generate_idea_list", generate_idea_list)
    graph.add_node("analyze_trends", analyze_ideas_with_trends)
    graph.add_node("parallel_analysis_2", parallel_analysis)
    graph.add_node("combine_and_score", combine_and_score)
    graph.add_node("validate_and_select", validate_and_select)
    graph.add_node("store_results_to_file", store_results_to_file)

    graph.set_entry_point("get_trending_industries")

    graph.add_edge("get_trending_industries", "generate_idea_list")
    graph.add_edge("generate_idea_list", "analyze_trends")
    graph.add_edge("analyze_trends", "parallel_analysis_2")
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
    if config is None:
        config = AnalysisConfig(
            max_retries=2,
            batch_size=3,
            timeout=30,
            enable_caching=True,
            cache_ttl_minutes=60
        )

    # Update global analyzer config if applicable
    global analyzer
    analyzer.config = config

    initial_state = MarketResearchState(
        config=config,
        start_time=datetime.utcnow(),
    )

    graph = create_market_research_graph()
    market_research_agent = graph.compile()

    init_db()

    try:
        logger.info("Starting Market Research Agent...")
        result_state = await market_research_agent.ainvoke(initial_state)
        best_idea = getattr(result_state, "best_business_idea", {}) or {}

        if best_idea:
            return {"best_business_idea": best_idea}
        else:
            return {"best_business_idea": None}

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {"error": str(e)}


async def get_or_generate_market_research_idea(state: Optional[MarketResearchState] = None, hours=24) -> Dict[str, Any]:
    init_db()

    if state is None:
        state = MarketResearchState(config=AnalysisConfig(), start_time=datetime.utcnow())

    if getattr(state, "user_idea", None):
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
        return {"best_business_idea": getattr(final_state, "best_business_idea", {})}

    # No user idea - check DB for recent idea
    ideas = load_ideas_from_db(limit=1)
    if ideas:
        idea = ideas[0]
        created_at_str = idea.get("created_at")
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str)
            if datetime.utcnow() - created_at < timedelta(hours=hours):
                logger.info("Using cached idea from DB (within last 24 hours).")
                return {"best_business_idea": idea}

    logger.info("No recent idea found. Running full market research agent...")
    result = await run_market_research_agent(state.config)
    logger.info(f"Result from run_market_research_agent: {result}")

    if result and "best_business_idea" in result and result["best_business_idea"]:
        best_idea = result["best_business_idea"]
        logger.info(f"Best idea before saving: {best_idea}")
        best_idea["created_at"] = datetime.utcnow().isoformat()
        save_idea_to_db(best_idea)
        return result

    return {"best_business_idea": None}


if __name__ == "__main__":
    idea = asyncio.run(get_or_generate_market_research_idea())
    print("Best Business Idea:", idea)